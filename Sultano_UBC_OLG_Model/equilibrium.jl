module Equilibrium

export GESolverOptions,
       solve_Ae_fixedpoint_given_prices,
       implied_w_given_wguess,
       solve_w_Ae_given_r,
       capital_excess_given_r,
       solve_GE,
       with_opts,
       EqmResults,
       solve_and_pack,
       pack_results

import ..Aggregates
import ..Distribution
import ..Grids
import ..Inheritance
import ..Parameters
import ..Prices
import ..Production
import ..Household
import ..Progress

"""
    GESolverOptions

Container for numerical solver settings used by the GE routines.

This struct collects tolerances, iteration limits, bracket endpoints, and
robustness/verbosity flags. It is designed to be immutable and updated via
`with_opts`.
"""
Base.@kwdef struct GESolverOptions
    w_lo::Float64      = 0.1
    w_hi::Float64      = 50.0
    tol_w::Float64     = 1e-6
    maxit_w::Int       = 120
    log_bisect::Bool   = true

    λ_Ae::Float64       = 0.3
    tol_Ae::Float64     = 1e-8
    maxit_Ae::Int       = 200
    log_update_Ae::Bool = true

    λ_b::Float64     = 0.35
    tol_b::Float64   = 1e-3
    verbose_b::Bool  = false

    α_DBN::Float64     = 0.8
    tol_DBN::Float64   = 1e-8
    max_iter_DBN::Int  = 400

    r_lo::Float64      = 0.001
    r_hi::Float64      = 0.08

    r_init::Float64         = NaN
    r_bracket_mult::Float64 = 1.5

    tol_r::Float64     = 1e-6
    tol_K::Float64     = 1e-6
    maxit_r::Int       = 40
    log_bisect_r::Bool = true
    max_expand_r::Int  = 40

    verbose::Bool          = true
    verbose_Ae::Bool       = true
    tol_w_bracket::Float64 = 1e-10

    λ_w_inner::Float64    = 0.5     # step size for wage fixed-point update (log-space)

    r_max::Float64            = 1.0
    accept_w_relerr::Float64  = 0.10
    return_best_on_fail::Bool = true
end

"""
    opts_namedtuple(opts::GESolverOptions)

Internal helper: convert a `GESolverOptions` instance into a `NamedTuple`
with the same fieldnames and values.
"""
@inline function opts_namedtuple(opts::GESolverOptions)
    names = fieldnames(GESolverOptions)
    vals  = Tuple(getfield(opts, nm) for nm in names)
    return NamedTuple{names}(vals)
end

"""
    with_opts(opts::GESolverOptions; kwargs...)

Return a copy of `opts` with any fields provided as keyword arguments replaced.

This is the canonical way to modify solver options without repeating the full
constructor.
"""
@inline function with_opts(opts::GESolverOptions; kwargs...)
    base = opts_namedtuple(opts)
    return GESolverOptions(; base..., kwargs...)
end

"""
    solve_Ae_fixedpoint_given_prices(p, g, prices0; bip, Ae0, b0, init_DBN, opts)

Solve the joint fixed point in `(A_e, b_pension)` for *given* prices `(r,w)`.

Algorithm:
- Iterates on entrepreneurial TFP `A_e` implied by the intermediate-good price identity.
- Simultaneously updates pension benefit `b_pension` toward the balanced-budget implied benefit.
- For each iterate, solves the household problem, computes the stationary distribution, and aggregates.

Returns:
`(p_final, V, pol, DBN, agg, prod, info)` where `p_final` contains the converged `(A_e, b_pension)`.
"""
function solve_Ae_fixedpoint_given_prices(p::Parameters.ModelParameters,
                                          g::Grids.ModelGrids,
                                          prices0::Prices.ModelPrices;
        bip::Inheritance.BequestInheritanceParams = Inheritance.BequestInheritanceParams(),
        Ae0::Float64 = p.A_e,
        b0::Float64  = p.b_pension,
        init_DBN = nothing,
        opts::GESolverOptions = GESolverOptions())

    Ae = Ae0
    if !(Ae > 0.0) || !isfinite(Ae)
        error("Initial Ae0 must be positive and finite. Got Ae0=$Ae0")
    end

    b = b0
    if !isfinite(b) || b < 0.0
        b = 0.0
    end

    V    = nothing
    pol  = nothing
    DBN  = init_DBN
    agg  = nothing
    prod = nothing

    Ae_implied = NaN
    b_implied  = NaN
    it_used    = 0

    pb_ae = (opts.verbose && opts.verbose_Ae) ? Progress.ProgressBar("Ae fixed-point", opts.maxit_Ae; show_eta=true) : nothing

    for it in 1:opts.maxit_Ae
        it_used = it
        (opts.verbose && opts.verbose_Ae) && Progress.update!(pb_ae, it)

        p_it = Parameters.with_Ae(p, Ae)
        p_it = Parameters.with_pension(p_it; τ_pension=p_it.τ_pension, b_pension=b)

        V, pol = Household.solve_household(p_it, g, prices0; bip=bip)

        DBN = Distribution.stationary_distribution(
            p_it, g, pol;
            bip=bip,
            init_DBN=DBN,
            max_iter=opts.max_iter_DBN,
            tol=opts.tol_DBN,
            α=opts.α_DBN
        )

        agg  = Aggregates.compute_aggregates(p_it, g, pol, DBN, prices0)
        prod = Production.final_good_block(p_it, agg)

        Ae_implied = Production.implied_Ae(prod.pQ, agg.Q, p_it.μ)
        Ae_floor = 1e-14
        Ae_implied = max(Ae_floor, Ae_implied)

        if !(Ae_implied > 0.0) || !isfinite(Ae_implied)
            error("A_e implied is not positive/finite: Ae_implied=$Ae_implied. " *
                  "Check Q=$(agg.Q), L=$(agg.L), pQ=$(prod.pQ).")
        end

        b_implied = getproperty(agg, :b_pension_implied)
        if !isfinite(b_implied) || b_implied < 0.0
            b_implied = 0.0
        end

        err_Ae = abs(Ae_implied - Ae) / max(1.0, abs(Ae))
        err_b  = abs(b_implied  - b)  / max(1.0, abs(b))

        if opts.verbose && opts.verbose_Ae
            println("FP it=$it  Ae=$Ae  implied=$Ae_implied  relerr_Ae=$err_Ae   b=$b  b_implied=$b_implied  relerr_b=$err_b")
            flush(stdout)
        elseif opts.verbose && opts.verbose_b
            println("b-FP it=$it  b=$b  implied=$b_implied  relerr_b=$err_b")
            flush(stdout)
        end

        if (err_Ae < opts.tol_Ae) && (err_b < opts.tol_b)
            Ae = Ae_implied
            b  = b_implied
            p_final = Parameters.with_pension(Parameters.with_Ae(p, Ae); τ_pension=p.τ_pension, b_pension=b)
            (opts.verbose && opts.verbose_Ae) && Progress.finish!(pb_ae, "converged")
            info = (converged=true, it=it_used, Ae=Ae, Ae_implied=Ae_implied, relerr_Ae=err_Ae,
                    b_pension=b, b_implied=b_implied, relerr_b=err_b)
            return p_final, V, pol, DBN, agg, prod, info
        end

        if opts.log_update_Ae
            Δ = log(Ae_implied) - log(Ae)
            Δ = clamp(Δ, -2.0, 2.0)
            Ae = exp(log(Ae) + opts.λ_Ae * Δ)
        else
            Ae = (1 - opts.λ_Ae) * Ae + opts.λ_Ae * Ae_implied
        end

        b = (1 - opts.λ_b) * b + opts.λ_b * b_implied
        b = max(0.0, b)
    end

    (opts.verbose && opts.verbose_Ae) && Progress.finish!(pb_ae, "max iterations")
    p_final = Parameters.with_pension(Parameters.with_Ae(p, Ae); τ_pension=p.τ_pension, b_pension=b)
    err_Ae = abs(Ae_implied - Ae) / max(1.0, abs(Ae))
    err_b  = abs(b_implied  - b)  / max(1.0, abs(b))
    info = (converged=false, it=it_used, Ae=Ae, Ae_implied=Ae_implied, relerr_Ae=err_Ae,
            b_pension=b, b_implied=b_implied, relerr_b=err_b)
    return p_final, V, pol, DBN, agg, prod, info
end

"""
    implied_w_given_wguess(p, g, r, w; bip, Ae0, init_DBN, opts)

Evaluate the wage fixed-point residual at a candidate wage `w` for a given `r`.

Steps:
- Construct prices `(r,w)`.
- Solve the inner `(A_e, b_pension)` fixed point for these prices.
- Compute the implied wage from the final-good block.

Returns:
`(w_implied, p_fp, V, pol, DBN, agg, prod, info_FP)`.
"""
function implied_w_given_wguess(p::Parameters.ModelParameters,
                                g::Grids.ModelGrids,
                                r::Float64,
                                w::Float64;
        bip::Inheritance.BequestInheritanceParams = Inheritance.BequestInheritanceParams(),
        Ae0::Float64 = p.A_e,
        init_DBN = nothing,
        opts::GESolverOptions = GESolverOptions())

    if !(w > 0.0) || !isfinite(w)
        error("w must be positive and finite. Got w=$w")
    end

    prices = Prices.ModelPrices(r, w)

    p_fp, V, pol, DBN, agg, prod, info_FP =
        solve_Ae_fixedpoint_given_prices(p, g, prices;
            bip=bip,
            Ae0=Ae0,
            b0=p.b_pension,
            init_DBN=init_DBN,
            opts=opts
        )

    return prod.w, p_fp, V, pol, DBN, agg, prod, info_FP
end

"""
    solve_w_Ae_given_r(p, g, r; bip, Ae0, init_DBN, opts)

Inner loop for a given interest rate `r`:
solve for the equilibrium wage `w` and the joint fixed point `(A_e, b_pension)`.

Uses bisection (optionally in logs) on the wage residual `w_implied(w) - w`.

Returns:
`(p_fp, prices_fp, V, pol, DBN, agg, prod, info)`.
"""
function solve_w_Ae_given_r(p::Parameters.ModelParameters,
                            g::Grids.ModelGrids,
                            r::Float64;
        bip::Inheritance.BequestInheritanceParams = Inheritance.BequestInheritanceParams(),
        Ae0::Float64 = p.A_e,
        init_DBN = nothing,
        opts::GESolverOptions = GESolverOptions())

    w_lo = opts.w_lo
    w_hi = opts.w_hi

    if !(w_lo > 0.0) || !(w_hi > 0.0) || !(w_lo < w_hi)
        error("Require 0 < w_lo < w_hi. Got w_lo=$w_lo, w_hi=$w_hi")
    end

    # Initial wage: geometric mean of [w_lo, w_hi] (observed equilibrium is ~0.5)
    w = exp(0.5 * (log(w_lo) + log(w_hi)))

    # Warm state carried across FP steps — consecutive w values are close, so
    # the Ae fixed-point re-converges in ~5–10 iterations instead of ~100.
    Ae_ws      = Ae0
    DBN_ws     = init_DBN
    p_ws_local = p

    # Best-tracking safety net: return lowest-|f| point if max iterations reached.
    best      = nothing
    best_absf = Inf
    best_wImp = NaN
    best_f    = NaN

    opts.verbose && (println("w-FP r=$r: start w=$w"); flush(stdout))

    pb_w = opts.verbose ? Progress.ProgressBar("w fixed-point (r=$r)", opts.maxit_w; show_eta=true) : nothing

    for it in 1:opts.maxit_w
        opts.verbose && Progress.update!(pb_w, it)
        wImp, pX, VX, polX, DBNX, aggX, prodX, infoFP =
            implied_w_given_wguess(p_ws_local, g, r, w;
                bip=bip, Ae0=Ae_ws, init_DBN=DBN_ws, opts=opts
            )

        fM = wImp - w

        if abs(fM) < best_absf
            best_absf = abs(fM)
            best      = (pX, Prices.ModelPrices(r, w), VX, polX, DBNX, aggX, prodX, infoFP)
            best_wImp = wImp
            best_f    = fM
        end

        if abs(fM) < opts.tol_w
            opts.verbose && Progress.finish!(pb_w, "converged")
            opts.verbose && (println("w-FP r=$r: converged it=$it w=$w |f|=$(abs(fM))"); flush(stdout))
            p_fp, prices_fp, V, pol, DBN, agg, prod, _ = best
            info = (converged=true, it=it, r=r, w=prices_fp.w, w_implied=wImp, f=fM, info_FP=infoFP)
            return p_fp, prices_fp, V, pol, DBN, agg, prod, info
        end

        # Carry warm state: next FP step starts from a very close w → fast Ae convergence.
        p_ws_local = pX
        Ae_ws      = pX.A_e
        DBN_ws     = DBNX

        # Log-space damped update: w ← w · (wImp/w)^λ_w_inner, clamp to [w_lo, w_hi].
        # The sign-change guarantee (from prior use of bisection) ensures η = d(log wImp)/d(log w) < 1
        # near w*, so this iteration contracts and converges.
        δ = clamp(log(wImp) - log(w), -2.0, 2.0)
        w = clamp(exp(log(w) + opts.λ_w_inner * δ), w_lo, w_hi)
    end

    # Max iterations reached — return best point found (same safety behaviour as bisection).
    opts.verbose && Progress.finish!(pb_w, "max iterations")
    p_fp, prices_fp, V, pol, DBN, agg, prod, infoFP = best
    info = (converged=false, it=opts.maxit_w, r=r, w=prices_fp.w,
            w_implied=best_wImp, f=best_f, info_FP=infoFP)
    return p_fp, prices_fp, V, pol, DBN, agg, prod, info
end

"""
    capital_excess_given_r(p, g, r; bip, Ae0, init_DBN, opts)

Compute the capital-market excess supply for a given interest rate `r`.

Steps:
- Solve the inner wage and `(A_e, b_pension)` loops for this `r`.
- Return `K_supply - K_demand` from aggregates.

Returns:
`(excessK, p_fp, prices_fp, V, pol, DBN, agg, prod, info_w)`.
"""
function capital_excess_given_r(p::Parameters.ModelParameters,
                                g::Grids.ModelGrids,
                                r::Float64;
        bip::Inheritance.BequestInheritanceParams = Inheritance.BequestInheritanceParams(),
        Ae0::Float64 = p.A_e,
        init_DBN = nothing,
        opts::GESolverOptions = GESolverOptions())

    p_fp, prices_fp, V, pol, DBN, agg, prod, info_w =
        solve_w_Ae_given_r(p, g, r;
            bip=bip,
            Ae0=Ae0,
            init_DBN=init_DBN,
            opts=opts
        )

    if !info_w.converged
        f = get(info_w, :f, NaN)
        w = get(info_w, :w, NaN)
        rel = abs(f) / max(1e-12, abs(w))

        if rel <= opts.accept_w_relerr
            opts.verbose && println("WARNING: wage solver not converged at r=$r (w=$w, f=$f, rel=$rel). Using best point.")
            flush(stdout)
        else
            error("Inner wage solver did not converge at r=$r. " *
                  "Last w=$w, w_implied=$(get(info_w, :w_implied, NaN)), f=$f.")
        end
    end

    excessK = agg.K_supply - agg.K_demand
    return excessK, p_fp, prices_fp, V, pol, DBN, agg, prod, info_w
end

"""
    EqmResults

Container for a solved steady state (GE) outcome.
"""
struct EqmResults
    p::Parameters.ModelParameters
    g::Grids.ModelGrids
    bip::Inheritance.BequestInheritanceParams
    prices::Prices.ModelPrices
    V::Array{Float64,4}
    pol::Household.Policies
    DBN::Array{Float64,4}
    agg
    prod
    info
end

"""
    pack_results(p, g, bip, prices, V, pol, DBN, agg, prod, info)

Convenience constructor for `EqmResults`.
"""
pack_results(p::Parameters.ModelParameters,
             g::Grids.ModelGrids,
             bip::Inheritance.BequestInheritanceParams,
             prices::Prices.ModelPrices,
             V::Array{Float64,4},
             pol::Household.Policies,
             DBN::Array{Float64,4},
             agg,
             prod,
             info) = EqmResults(p, g, bip, prices, V, pol, DBN, agg, prod, info)

"""
    solve_and_pack(p, g; bip=..., init_DBN=nothing, opts=GESolverOptions())

Solve for GE and return an `EqmResults` object.
"""
function solve_and_pack(p::Parameters.ModelParameters,
                        g::Grids.ModelGrids;
                        bip::Inheritance.BequestInheritanceParams = Inheritance.BequestInheritanceParams(),
                        init_DBN = nothing,
                        opts::GESolverOptions = GESolverOptions())
    p_fp, prices_fp, V, pol, DBN, agg, prod, info =
        solve_GE(p, g; bip=bip, init_DBN=init_DBN, opts=opts)
    return pack_results(p_fp, g, bip, prices_fp, V, pol, DBN, agg, prod, info)
end

"""
    solve_GE(p, g; bip=..., init_DBN=nothing, opts=GESolverOptions())

Solve the general equilibrium steady state.

Outer loop:
- finds `r` that clears the capital market via bisection/bracketing.

Inner loop (for each r):
- finds `w` that satisfies the wage fixed point and solves the joint `(A_e, b_pension)` fixed point.

Returns:
`(p_fp, prices_fp, V, pol, DBN, agg, prod, info)` where `info` contains convergence diagnostics.
"""
function solve_GE(p::Parameters.ModelParameters,
                  g::Grids.ModelGrids;
        bip::Inheritance.BequestInheritanceParams = Inheritance.BequestInheritanceParams(),
        init_DBN = nothing,
        opts::GESolverOptions = GESolverOptions())

    r_min = -p.δ + 1e-8

    rL0 = max(opts.r_lo, r_min)
    rH0 = max(opts.r_hi, r_min + 1e-6)
    rL0 < rH0 || error("Require r_lo < r_hi. Got r_lo=$(opts.r_lo), r_hi=$(opts.r_hi)")

    p_ws   = p
    DBN_ws = init_DBN

    if DBN_ws !== nothing
        H, na, nz, _ = size(DBN_ws)
        if (H != p.MaxAge) || (na != p.na) || (nz != p.nz)
            opts.verbose && println("solve_GE: init_DBN has wrong shape; ignoring warm start.")
            DBN_ws = nothing
        end
    end

    best = nothing
    best_absf = Inf

    function update_best!(f, p1, prices1, V1, pol1, DBN1, agg1, prod1, info1)
        af = abs(f)
        if af < best_absf
            best_absf = af
            best = (p1, prices1, V1, pol1, DBN1, agg1, prod1, info1, f)
        end
        return nothing
    end

    function safe_excessK(r::Float64, p_ws_local, DBN_ws_local)
        try
            f, p1, prices1, V1, pol1, DBN1, agg1, prod1, info1 =
                capital_excess_given_r(p_ws_local, g, r;
                    bip=bip,
                    Ae0=p_ws_local.A_e,
                    init_DBN=DBN_ws_local,
                    opts=opts
                )
            return true, f, p1, prices1, V1, pol1, DBN1, agg1, prod1, info1
        catch err
            if opts.verbose
                println("GE infeasible at r=$r because: ", sprint(showerror, err))
                flush(stdout)
            end
            return false, err, r
        end
    end

    r0 = if isfinite(opts.r_init) && (opts.r_init > r_min)
        clamp(opts.r_init, rL0, rH0)
    elseif rL0 > 0.0 && rH0 > 0.0
        exp(0.5 * (log(rL0) + log(rH0)))
    else
        0.5 * (rL0 + rH0)
    end

    start = safe_excessK(r0, p_ws, DBN_ws)

    if start[1] !== true
        opts.verbose && println("GE: r0=$r0 infeasible. Scanning for feasible r...")
        rs = (rL0 > 0.0 && rH0 > 0.0) ? exp.(range(log(rL0), log(rH0), length=12)) :
                                       collect(range(rL0, rH0, length=12))
        found = false
        for rr in rs
            tmp = safe_excessK(rr, p_ws, DBN_ws)
            if tmp[1] === true
                start = tmp
                r0 = rr
                found = true
                break
            end
        end
        found || error("solve_GE: could not find ANY feasible r in [r_lo,r_hi].")
    end

    _, f0, p0, prices0, V0, pol0, DBN0, agg0, prod0, info0 = start
    p_ws   = p0
    DBN_ws = DBN0

    best = (p0, prices0, V0, pol0, DBN0, agg0, prod0, info0, f0)
    best_absf = abs(f0)

    opts.verbose && println("\nGE start: r0=$r0  excessK=$f0  (Ks=$(agg0.K_supply), Kd=$(agg0.K_demand))")

    γ = 1.35

    rA = r0; fA = f0; payloadA = best; foundA = false
    rr = r0
    p_tmp, DBN_tmp = p_ws, DBN_ws
    for _ in 1:opts.max_expand_r
        rr = max(r_min, rr / γ)
        tmp = safe_excessK(rr, p_tmp, DBN_tmp)
        tmp[1] === true || continue
        _, f, p1, prices1, V1, pol1, DBN1, agg1, prod1, info1 = tmp
        p_tmp, DBN_tmp = p1, DBN1
        update_best!(f, p1, prices1, V1, pol1, DBN1, agg1, prod1, info1)
        payloadA = (p1, prices1, V1, pol1, DBN1, agg1, prod1, info1, f)
        rA, fA = rr, f
        if sign(fA) != sign(f0)
            foundA = true
            break
        end
    end

    # Only expand UP if DOWN direction did not find a bracket.
    # With r_init=0.06 > r* and f0>0, the foundA (DOWN) loop always finds the bracket
    # in 2–4 steps; running foundB (UP) unconditionally wastes ~9 r-evaluations per GE
    # solve (~22 h on a MacBook Air with na=100), multiplied across all calibration calls.
    rB = r0; fB = f0; payloadB = best; foundB = false
    if !foundA
        rr = r0
        p_tmp, DBN_tmp = p_ws, DBN_ws
        for _ in 1:opts.max_expand_r
            rr = rr * γ
            rr > opts.r_max && break
            tmp = safe_excessK(rr, p_tmp, DBN_tmp)
            tmp[1] === true || continue
            _, f, p1, prices1, V1, pol1, DBN1, agg1, prod1, info1 = tmp
            p_tmp, DBN_tmp = p1, DBN1
            update_best!(f, p1, prices1, V1, pol1, DBN1, agg1, prod1, info1)
            payloadB = (p1, prices1, V1, pol1, DBN1, agg1, prod1, info1, f)
            rB, fB = rr, f
            if sign(fB) != sign(f0)
                foundB = true
                break
            end
        end
    end

    if foundA
        r_lo, f_lo = rA, fA
        r_hi, f_hi = r0, f0
        p_ws   = payloadA[1]
        DBN_ws = payloadA[5]
    elseif foundB
        r_lo, f_lo = r0, f0
        r_hi, f_hi = rB, fB
        p_ws   = payloadB[1]
        DBN_ws = payloadB[5]
    else
        msg = "solve_GE: could not bracket capital market root using feasible r evaluations."
        if opts.return_best_on_fail
            opts.verbose && (println(msg * " Returning best feasible point (converged=false)."); flush(stdout))
            p_fp, prices_fp, V, pol, DBN, agg, prod, info_w, f = best
            info = (converged=false, it=0, r=prices_fp.r, w=prices_fp.w,
                    excessK=f, K_supply=agg.K_supply, K_demand=agg.K_demand,
                    failed="no_r_bracket", info_inner=info_w)
            return p_fp, prices_fp, V, pol, DBN, agg, prod, info
        else
            error(msg * " Try widening r_lo/r_hi or inspect K_supply - K_demand at a few r values.")
        end
    end

    opts.verbose && println("GE bracket: r_lo=$r_lo (f=$f_lo), r_hi=$r_hi (f=$f_hi)")

    pb_r = opts.verbose ? Progress.ProgressBar("r-bisection (GE)", opts.maxit_r; show_eta=true) : nothing

    for it in 1:opts.maxit_r
        opts.verbose && Progress.update!(pb_r, it)
        rM = (opts.log_bisect_r && r_lo > 0.0 && r_hi > 0.0) ? exp(0.5 * (log(r_lo) + log(r_hi))) : 0.5 * (r_lo + r_hi)

        tmp = safe_excessK(rM, p_ws, DBN_ws)
        if tmp[1] !== true
            opts.verbose && println("r midpoint infeasible at r=$rM; shrinking bracket.")
            if (r_hi - rM) > (rM - r_lo)
                r_hi = rM
            else
                r_lo = rM
            end
            continue
        end

        _, fM, pM, pricesM, VM, polM, DBNM, aggM, prodM, info_wM = tmp
        update_best!(fM, pM, pricesM, VM, polM, DBNM, aggM, prodM, info_wM)

        Krel = abs(fM) / max(1e-12, abs(aggM.K_supply))
        rrel = abs(r_hi - r_lo) / max(1e-12, abs(rM))

        if opts.verbose
            println("r-bisect it=$it  r=$rM  w=$(pricesM.w)  Ae=$(pM.A_e)  b=$(pM.b_pension)  excessK=$fM  Krel=$Krel  [Ks=$(aggM.K_supply) Kd=$(aggM.K_demand)]")
            flush(stdout)
        end

        if (Krel < opts.tol_K) || (rrel < opts.tol_r)
            opts.verbose && Progress.finish!(pb_r, "converged")
            p_fp, prices_fp, V, pol, DBN, agg, prod, info_w, f = best
            info = (converged=true, it=it, r=prices_fp.r, w=prices_fp.w,
                    excessK=f, K_supply=agg.K_supply, K_demand=agg.K_demand,
                    info_inner=info_w)
            return p_fp, prices_fp, V, pol, DBN, agg, prod, info
        end

        if sign(fM) == sign(f_lo)
            r_lo, f_lo = rM, fM
        else
            r_hi, f_hi = rM, fM
        end

        # Warm-start the NEXT bisection step from the best point found so far,
        # not from pM.  Resetting to best prevents Ae from drifting monotonically
        # when tol_Ae is loose: without this, all midpoints give the same sign of
        # excessK (one-sided bisection), wasting ~11 extra r-evaluations per GE solve.
        p_fp_best, _, _, _, DBN_best, _, _, _, _ = best
        p_ws   = p_fp_best
        DBN_ws = DBN_best
    end

    opts.verbose && Progress.finish!(pb_r, "max iterations")
    p_fp, prices_fp, V, pol, DBN, agg, prod, info_w, f = best
    info = (converged=false, it=opts.maxit_r, r=prices_fp.r, w=prices_fp.w,
            excessK=f, K_supply=agg.K_supply, K_demand=agg.K_demand,
            failed="maxit_r", info_inner=info_w)
    return p_fp, prices_fp, V, pol, DBN, agg, prod, info
end

end