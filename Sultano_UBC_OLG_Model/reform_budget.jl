module ReformBudget

export solve_reform_balanced_tau

using Printf

import ..Inheritance
import ..Equilibrium
import ..Reporting

"""
    WarmStart

Mutable container for warm-start objects used when repeatedly solving GE
across different reform parameters.

Fields
- `r`: last solved interest rate (used to initialize the next GE solve)
- `DBN`: last solved stationary distribution (used to initialize the next DBN iteration)
"""
mutable struct WarmStart
    r::Float64
    DBN
end

"""
    WarmStart()

Construct an empty warm start with `r = NaN` and `DBN = nothing`.
"""
WarmStart() = WarmStart(NaN, nothing)

"""
    solve_reform_balanced_tau(p, g;
        T,
        exemption=0.0,
        τ_lo=0.0,
        τ_hi=0.6,
        tol_rel=1e-6,
        maxit=40,
        opts_ge=Equilibrium.GESolverOptions(),
        init_res=nothing,
        verbose=true)

Solve a balanced-budget reform financed by an inheritance tax.

Policy environment (steady state):
- Newborn transfer: `transfer = T` in `Inheritance.BequestInheritanceParams`.
  In steady state, the mass of entrants equals the mass of deaths each period,
  so spending is `spend = T * deaths_mass`.

Budget balance:
- Choose `τ_b` by bisection such that inheritance tax revenue equals spending:
  `TaxRev_flow(τ_b) - spend = 0`.

Warm-starting:
- Across τ evaluations, warm-start GE with the last solved `r` and `DBN`.
- If `init_res` (baseline) is provided, initialize warm starts from it.

Returns a NamedTuple:
`(τ, res, taxrev, spend, entrants_mass, residual)`
"""
function solve_reform_balanced_tau(p, g;
    T::Float64,
    exemption::Float64 = 0.0,
    τ_lo::Float64 = 0.0,
    τ_hi::Float64 = 0.6,
    tol_rel::Float64 = 1e-6,
    maxit::Int = 40,
    opts_ge::Equilibrium.GESolverOptions = Equilibrium.GESolverOptions(),
    init_res = nothing,
    verbose::Bool = true
)
    warm = WarmStart()
    if init_res !== nothing
        warm.r   = init_res.prices.r
        warm.DBN = init_res.DBN
    else
        warm.r = opts_ge.r_init
    end

    """
        eval_tau(τ)

    Evaluate the balanced-budget residual at inheritance tax rate `τ`.

    Steps:
    - Solve GE with bequest/inheritance parameters `τ_b=τ`, `exemption`, `transfer=T`
      (warm-starting with the last `r` and `DBN`).
    - Compute inheritance tax revenue from `Reporting.bequest_stats(res)`.
    - Compute spending as `T * deaths_mass`.
    - Return residual `taxrev - spend`.

    Returns:
    `(res, resid, taxrev, spend, entrants_mass)`.
    """
    function eval_tau(τ::Float64)
        bip = Inheritance.BequestInheritanceParams(τ_b=τ, exemption=exemption, transfer=T)

        opts2 = isfinite(warm.r) ?
            Equilibrium.with_opts(opts_ge; r_init=warm.r, verbose=false) :
            Equilibrium.with_opts(opts_ge; r_init=opts_ge.r_init, verbose=false)

        res = Equilibrium.solve_and_pack(p, g; bip=bip, init_DBN=warm.DBN, opts=opts2)

        warm.r   = res.prices.r
        warm.DBN = res.DBN

        bq = Reporting.bequest_stats(res)
        taxrev        = bq.TaxRev_flow
        entrants_mass = bq.deaths_mass
        spend         = T * entrants_mass

        resid = taxrev - spend
        return res, resid, taxrev, spend, entrants_mass
    end

    resL, fL, revL, spL, mL = eval_tau(τ_lo)
    resH, fH, revH, spH, mH = eval_tau(τ_hi)

    tries = 0
    while (isfinite(fL) && isfinite(fH) && sign(fL) == sign(fH)) && τ_hi < 0.99 && tries < 12
        τ_hi = min(0.99, 1.25 * τ_hi + 0.02)
        resH, fH, revH, spH, mH = eval_tau(τ_hi)
        tries += 1
    end

    if !(isfinite(fL) && isfinite(fH)) || sign(fL) == sign(fH)
        error("Could not bracket balanced-budget τ_b. Maybe T is too high/infeasible, or τ_hi too low. " *
              "f(τ_lo)=$fL, f(τ_hi)=$fH with τ_hi=$τ_hi.")
    end

    best = (res = abs(fL) < abs(fH) ? resL : resH,
            τ   = abs(fL) < abs(fH) ? τ_lo : τ_hi,
            f   = min(abs(fL), abs(fH)),
            taxrev = abs(fL) < abs(fH) ? revL : revH,
            spend  = abs(fL) < abs(fH) ? spL  : spH,
            entrants_mass = abs(fL) < abs(fH) ? mL : mH)

    for it in 1:maxit
        τ_mid = 0.5 * (τ_lo + τ_hi)
        resM, fM, revM, spM, mM = eval_tau(τ_mid)

        if !isfinite(fM)
            τ_try1 = 0.75 * τ_lo + 0.25 * τ_hi
            res1, f1, rev1, sp1, m1 = eval_tau(τ_try1)
            if isfinite(f1)
                τ_mid, resM, fM, revM, spM, mM = τ_try1, res1, f1, rev1, sp1, m1
            else
                τ_try2 = 0.25 * τ_lo + 0.75 * τ_hi
                res2, f2, rev2, sp2, m2 = eval_tau(τ_try2)
                if isfinite(f2)
                    τ_mid, resM, fM, revM, spM, mM = τ_try2, res2, f2, rev2, sp2, m2
                else
                    τ_hi = τ_mid
                    continue
                end
            end
        end

        rel = abs(fM) / max(1e-12, abs(spM))
        if verbose
            @printf("  it=%d  τ=%.6f  taxrev=%.6f  spend=%.6f  relgap=%.3e\n",
                    it, τ_mid, revM, spM, rel)
        end

        if abs(fM) < best.f
            best = (res=resM, τ=τ_mid, f=abs(fM), taxrev=revM, spend=spM, entrants_mass=mM)
        end

        if rel < tol_rel
            return (τ=τ_mid, res=resM, taxrev=revM, spend=spM, entrants_mass=mM, residual=fM)
        end

        if sign(fM) == sign(fL)
            τ_lo = τ_mid
            fL   = fM
        else
            τ_hi = τ_mid
            fH   = fM
        end
    end

    return (τ=best.τ, res=best.res, taxrev=best.taxrev, spend=best.spend, entrants_mass=best.entrants_mass, residual=NaN)
end

end