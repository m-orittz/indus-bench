module Calibration

export CalibrationTargets, CalibrationOptions, CalibrationResult,
       compute_moments, calibrate,
       save_calibration, load_calibration

using LinearAlgebra
using Printf
using Serialization

import ..Parameters
import ..Grids
import ..Equilibrium
import ..Reporting
import ..Inheritance

# ============================================================
# Public structs
# ============================================================

"""
    CalibrationTargets

Empirical targets for the three jointly calibrated moments.

Fields
- `ent_wealth_share`: entrepreneurs' share of total population wealth (default 0.416)
- `ent_pop_share`:    entrepreneurs' share of total population        (default 0.115)
- `debt_to_Y`:        business debt / output ratio                    (default 1.52)
"""
Base.@kwdef struct CalibrationTargets
    ent_wealth_share::Float64 = 0.416
    ent_pop_share::Float64    = 0.115
    debt_to_Y::Float64        = 1.52
end

"""
    CalibrationOptions

Numerical settings for the outer calibration loop.

Fields
- `tol`:              convergence threshold on max absolute normalised residual (default 1e-4).
                      A value of 1e-4 means each moment is matched to within 0.01 % of its
                      target — tighter than any empirical measurement uncertainty in the targets,
                      so this is adequate for a final thesis run.
- `max_iter`:         maximum Broyden iterations (default 50)
- `h_fd`:             relative step for the initial finite-difference Jacobian (default 0.03).
                      3 % is a good balance: large enough to avoid floating-point cancellation,
                      small enough that nonlinearity does not bias the Jacobian estimate.
- `parallel_jacobian`: run the 3 Jacobian columns concurrently with `Threads.@spawn`
                      (default true). Effective when Julia is started with multiple threads
                      (`julia --threads N`). Has no effect if `Threads.nthreads() == 1`.
- `min_z_scale`:      lower bound for z_scale  (default 0.1)
- `min_F`:            lower bound for F        (default 0.0)
- `min_θ_scalar`:     lower bound for θ_scalar (default 0.1)
- `verbose`:          print iteration diagnostics (default true)
- `ge_opts`:          `GESolverOptions` forwarded to every inner GE solve;
                      verbose flags default to false to suppress inner-loop output.
                      The inner-loop defaults (tol_K=1e-6, tol_r=1e-6, tol_Ae=1e-8,
                      tol_DBN=1e-8) are already thesis-grade tight.
"""
Base.@kwdef struct CalibrationOptions
    tol::Float64               = 1e-4
    max_iter::Int              = 50
    h_fd::Float64              = 0.03
    parallel_jacobian::Bool    = true
    min_z_scale::Float64       = 0.1
    min_F::Float64             = 0.0
    min_θ_scalar::Float64      = 0.1
    verbose::Bool              = true
    ge_opts::Equilibrium.GESolverOptions =
        Equilibrium.GESolverOptions(verbose=false, verbose_Ae=false)
end

"""
    CalibrationResult

Output from `calibrate(...)`.

Fields
- `z_scale`:   calibrated z_scale parameter
- `F`:         calibrated entry cost
- `θ_scalar`:  calibrated scalar applied to the baseline θ_vec
- `moments`:   NamedTuple of achieved moments at the solution
- `residuals`: vector of normalised residuals `(model - target)/target` (length 3)
- `converged`: whether the tolerance was met
- `iter`:      number of Broyden iterations used
- `res`:       `EqmResults` from the final GE solve
"""
struct CalibrationResult
    z_scale::Float64
    F::Float64
    θ_scalar::Float64
    moments::NamedTuple
    residuals::Vector{Float64}
    converged::Bool
    iter::Int
    res::Equilibrium.EqmResults
end

# ============================================================
# Public moment-extraction function
# ============================================================

"""
    compute_moments(res)

Extract the three calibration moments from a solved `EqmResults`:

1. `ent_wealth_share` — entrepreneurs' share of total population wealth
2. `ent_pop_share`    — entrepreneurs' share of total population
3. `debt_to_Y`        — business debt / output ratio

These are the moments targeted by `calibrate`.
"""
function compute_moments(res::Equilibrium.EqmResults)
    ms = Reporting.macro_stats(res)

    ent_wealth_share = _ent_wealth_share_pop(res)

    return (
        ent_wealth_share = ent_wealth_share,
        ent_pop_share    = ms.Ent_share_pop,
        debt_to_Y        = ms.Debt_to_Y
    )
end

# ============================================================
# Main calibration routine
# ============================================================

"""
    calibrate(p_base; targets, opts, bip)

Jointly calibrate three structural parameters to three empirical targets:

| Parameter   | Targets                                     |
|-------------|---------------------------------------------|
| `z_scale`   | entrepreneur share of total population wealth |
| `F`         | entrepreneur share of total population        |
| `θ_scalar`  | business debt-to-output ratio               |

`θ_scalar` multiplies the *baseline* `p_base.θ_vec` component-wise; the
shape of the borrowing-constraint schedule is preserved.

The algorithm is **Broyden's quasi-Newton method**:
1. Evaluate the moment vector at the initial point (1 GE solve).
2. Estimate the 3×3 Jacobian via forward finite differences (3 GE solves).
3. Iterate with rank-1 Broyden updates and a simple backtracking line search.

Warm-starting (passing the previous DBN as `init_DBN`) is used at each
iteration to accelerate successive GE solves.

Returns a `CalibrationResult`.
"""
function calibrate(p_base::Parameters.ModelParameters;
                   targets::CalibrationTargets = CalibrationTargets(),
                   opts::CalibrationOptions    = CalibrationOptions(),
                   bip::Inheritance.BequestInheritanceParams =
                       Inheritance.BequestInheritanceParams(),
                   init_ge_file::Union{String,Nothing} = nothing,
                   force_reinit_ge::Bool = false)

    θ_vec_base = copy(p_base.θ_vec)

    # Initial search point: inherit z_scale and F from baseline, θ_scalar = 1
    x = _clamp_x([p_base.z_scale, max(p_base.F, 0.0), 1.0], opts)

    # ------------------------------------------------------------------
    # Inner helper: solve GE for parameter vector x and return (r, mom, res).
    #
    # Warm-start strategy:
    #   • Initial evaluation (step 1): no warm start (cold solve).
    #   • Jacobian columns (step 2): each column warm-starts from the *same*
    #     baseline res — safe because res.DBN is read-only inside eval_x.
    #   • Broyden iterations (step 3): each iteration warm-starts from the
    #     accepted point of the previous iteration.
    #   • Line-search trials: each trial warm-starts from the start of the
    #     current Broyden iteration.
    # ------------------------------------------------------------------
    function eval_x(xv::Vector{Float64}, warm_res)
        p_cal = Parameters.with(p_base;
            z_scale = xv[1],
            F       = xv[2],
            θ_vec   = θ_vec_base .* xv[3]
        )
        g_cal = Grids.build_grids(p_cal)

        # Pass previous DBN as initial distribution to speed up convergence.
        init_DBN = (warm_res !== nothing) ? warm_res.DBN : nothing

        res = Equilibrium.solve_and_pack(p_cal, g_cal;
            bip      = bip,
            init_DBN = init_DBN,
            opts     = opts.ge_opts
        )

        mom = compute_moments(res)
        r   = _residuals(mom, targets)
        return r, mom, res
    end

    # ------------------------------------------------------------------
    # Step 1: evaluate at initial point
    # ------------------------------------------------------------------
    if opts.verbose
        println("\n╔══════════════════════════════════════════╗")
        println(  "║         OLG CALIBRATION (Broyden)        ║")
        println(  "╚══════════════════════════════════════════╝")
        @printf("Targets:  ent_wealth_share=%.4f  ent_pop_share=%.4f  debt_to_Y=%.4f\n",
                targets.ent_wealth_share, targets.ent_pop_share, targets.debt_to_Y)
        @printf("Initial:  z_scale=%.4f  F=%.4f  θ_scalar=%.4f\n", x[1], x[2], x[3])
        flush(stdout)
    end

    # ------------------------------------------------------------------
    # Step 1: evaluate at initial point — load from cache if available.
    # ------------------------------------------------------------------
    use_cached = (init_ge_file !== nothing) && !force_reinit_ge && isfile(init_ge_file)

    r, mom, res = if use_cached
        opts.verbose && println("Loading cached initial GE from: $init_ge_file")
        flush(stdout)
        res0 = open(init_ge_file, "r") do io; deserialize(io); end
        mom0 = compute_moments(res0)
        r0   = _residuals(mom0, targets)
        r0, mom0, res0
    else
        opts.verbose && println("Evaluating initial point...")
        flush(stdout)
        eval_x(x, nothing)
    end

    if !use_cached && init_ge_file !== nothing
        open(init_ge_file, "w") do io; serialize(io, res); end
        opts.verbose && println("Initial GE saved to: $init_ge_file")
    end

    if opts.verbose
        _print_iter(0, x, r, mom, targets)
    end

    if maximum(abs, r) < opts.tol
        return CalibrationResult(x[1], x[2], x[3], mom, r, true, 0, res)
    end

    # ------------------------------------------------------------------
    # Step 2: estimate Jacobian via forward finite differences (3 GE solves).
    #
    # Each column perturbs one parameter independently, warm-started from the
    # same baseline `res`.  The 3 solves are independent so they can run
    # concurrently when opts.parallel_jacobian == true and nthreads() > 1.
    # ------------------------------------------------------------------
    use_parallel = opts.parallel_jacobian && Threads.nthreads() > 1
    if opts.verbose
        nsolves = use_parallel ? "3 GE solves in parallel on $(Threads.nthreads()) threads" :
                                 "3 GE solves sequentially"
        println("\nEstimating Jacobian ($nsolves)...")
        flush(stdout)
    end

    # Pre-compute perturbed points and actual step sizes before any spawning.
    xps  = Vector{Vector{Float64}}(undef, 3)
    hs   = Vector{Float64}(undef, 3)
    for j in 1:3
        xp    = copy(x)
        h_abs = opts.h_fd * max(abs(x[j]), 1e-2)
        xp[j] = x[j] + h_abs
        xp    = _clamp_x(xp, opts)
        hs[j] = xp[j] - x[j]
        xps[j] = xp
    end

    # Evaluate the 3 columns — in parallel or sequentially.
    J_cols = if use_parallel
        tasks = Vector{Task}(undef, 3)
        for j in 1:3
            xpj = xps[j]            # capture per-iteration value, not loop variable
            tasks[j] = Threads.@spawn eval_x(xpj, res)
        end
        [fetch(tasks[j]) for j in 1:3]
    else
        [eval_x(xps[j], res) for j in 1:3]
    end

    J = zeros(3, 3)
    for j in 1:3
        rp, _, _ = J_cols[j]
        J[:, j]  = (rp .- r) ./ hs[j]
        opts.verbose && @printf("  J col %d estimated  (h=%.2e)\n", j, hs[j])
    end
    opts.verbose && flush(stdout)

    # ------------------------------------------------------------------
    # Step 3: Broyden iterations
    # ------------------------------------------------------------------
    B = copy(J)   # Broyden approximate Jacobian  (∂r/∂x)

    for it in 1:opts.max_iter

        # --- Newton-like step: B * Δx = -r ---
        Δx = try
            -B \ r
        catch
            opts.verbose && println("  WARNING: Jacobian singular; falling back to gradient step.")
            -0.05 .* sign.(r)
        end

        # --- Backtracking line search ---
        α     = 1.0
        r_new = r       # will be overwritten
        mom_new = mom
        res_new = res

        for ls in 1:6
            x_try = _clamp_x(x .+ α .* Δx, opts)
            r_try, mom_try, res_try = eval_x(x_try, res)

            if maximum(abs, r_try) < maximum(abs, r) * (1.0 - 1e-4 * α)
                r_new   = r_try
                mom_new = mom_try
                res_new = res_try
                Δx      = x_try .- x   # actual step (after clamping)
                break
            end

            α *= 0.5
            # always accept on final attempt to keep moving
            if ls == 6
                r_new   = r_try
                mom_new = mom_try
                res_new = res_try
                Δx      = x_try .- x
            end
        end

        # --- Broyden rank-1 update: B ← B + (Δr − B Δs) Δsᵀ / (Δsᵀ Δs) ---
        Δs    = Δx
        Δr    = r_new .- r
        denom = dot(Δs, Δs)

        if denom > 1e-14
            B .+= ((Δr .- B * Δs) ./ denom) .* Δs'
        end

        x   = x .+ Δs
        r   = r_new
        res = res_new

        if opts.verbose
            _print_iter(it, x, r, mom_new, targets)
        end

        if maximum(abs, r) < opts.tol
            return CalibrationResult(x[1], x[2], x[3], mom_new, r, true, it, res)
        end
    end

    # Max iterations reached — return best available result
    mom_final = compute_moments(res)
    r_final   = _residuals(mom_final, targets)
    if opts.verbose
        println("\nWARNING: calibration reached max_iter=$(opts.max_iter) without converging.")
        @printf("Final ||r||_inf = %.2e  (tol = %.2e)\n", maximum(abs, r_final), opts.tol)
        flush(stdout)
    end

    return CalibrationResult(x[1], x[2], x[3], mom_final, r_final, false, opts.max_iter, res)
end

# ============================================================
# Internal helpers
# ============================================================

"""
    _ent_wealth_share_pop(res)

Compute the share of total population wealth held by entrepreneurs.
"""
function _ent_wealth_share_pop(res::Equilibrium.EqmResults)
    DBN = res.DBN
    pol = res.pol
    g   = res.g

    H, na, nz, ne = size(DBN)

    totW = 0.0
    entW = 0.0

    @inbounds for h in 1:H
        for ia in 1:na, iz in 1:nz, ep in 1:ne
            m = DBN[h, ia, iz, ep]
            m == 0.0 && continue
            a = g.agrid[ia]
            totW += a * m
            if pol.e[h, ia, iz, ep] == 1
                entW += a * m
            end
        end
    end

    return totW > 0.0 ? entW / totW : NaN
end

function _residuals(mom, tgt::CalibrationTargets)
    return Float64[
        (mom.ent_wealth_share - tgt.ent_wealth_share) / tgt.ent_wealth_share,
        (mom.ent_pop_share    - tgt.ent_pop_share)    / tgt.ent_pop_share,
        (mom.debt_to_Y        - tgt.debt_to_Y)        / tgt.debt_to_Y
    ]
end

function _clamp_x(x::Vector{Float64}, opts::CalibrationOptions)
    return Float64[
        max(x[1], opts.min_z_scale),
        max(x[2], opts.min_F),
        max(x[3], opts.min_θ_scalar)
    ]
end

function _print_iter(it::Int, x::Vector{Float64}, r::Vector{Float64}, mom,
                     tgt::CalibrationTargets)
    println()
    if it == 0
        @printf("--- iter 0 (initial evaluation) ---\n")
    else
        @printf("--- iter %d ---\n", it)
    end
    @printf("  z_scale  = %10.6f    θ_scalar = %10.6f    F = %10.4f\n", x[1], x[3], x[2])
    @printf("  ent_wealth_share = %.5f  target = %.5f  res = %+.4f\n", mom.ent_wealth_share, tgt.ent_wealth_share, r[1])
    @printf("  ent_pop_share    = %.5f  target = %.5f  res = %+.4f\n", mom.ent_pop_share,    tgt.ent_pop_share,    r[2])
    @printf("  debt_to_Y        = %.5f  target = %.5f  res = %+.4f\n", mom.debt_to_Y,        tgt.debt_to_Y,        r[3])
    @printf("  ||r||_inf = %.3e\n", maximum(abs, r))
    flush(stdout)
end

# ============================================================
# Persistence
# ============================================================

"""
    save_calibration(result, path)

Serialize a `CalibrationResult` to `path` using Julia's built-in
`Serialization` module (no external packages required).

The file stores the complete result, including:
- calibrated scalars (`z_scale`, `F`, `θ_scalar`)
- the full `EqmResults` baseline steady state (`res`)
- achieved moments and residuals

Convention: use the `.jls` extension (Julia serialization).

# Example
```julia
save_calibration(cal, "baseline_calibration.jls")
```
"""
function save_calibration(result::CalibrationResult, path::String)
    open(path, "w") do io
        serialize(io, result)
    end
    @printf("Calibration saved to: %s\n", path)
    return nothing
end

"""
    load_calibration(path) -> CalibrationResult

Deserialize a `CalibrationResult` previously saved with `save_calibration`.

The loaded result contains the calibrated parameters and the full baseline
`EqmResults`, ready to be passed to a reform module.

# Example
```julia
cal = load_calibration("baseline_calibration.jls")
p_cal = cal.res.p          # calibrated ModelParameters
res_baseline = cal.res     # baseline steady-state EqmResults
```
"""
function load_calibration(path::String)::CalibrationResult
    result = open(path, "r") do io
        deserialize(io)
    end
    if !(result isa CalibrationResult)
        error("File does not contain a CalibrationResult: got $(typeof(result))")
    end
    @printf("Calibration loaded from: %s\n", path)
    return result
end

end
