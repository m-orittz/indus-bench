# ==============================================================================
# main.jl — OLG Entrepreneurship Model  (MA Thesis)
#
# Structure
#   1. OLGModel parent module — wraps every submodule in dependency order.
#   2. Step 1: Calibration — run once and persist; reload on subsequent runs.
#   3. Step 2: Reforms — balanced-budget inheritance-tax experiments,
#              warm-started from the calibrated baseline.
#
# Usage
#   julia --threads auto main.jl
# ==============================================================================

module OLGModel
    include("demographics.jl")   # Demographics
    include("parameters.jl")     # Parameters   ← Demographics
    include("prices.jl")         # Prices
    include("grids.jl")          # Grids         ← Parameters
    include("utility.jl")        # Utility       ← Parameters
    include("production.jl")     # Production    ← Parameters, Prices, Utility
    include("inheritance.jl")    # Inheritance   ← Parameters, Grids
    include("progress.jl")      # Progress      ← standalone utility
    include("household.jl")      # Household     ← Parameters, Grids, Prices,
                                 #                  Utility, Production, Inheritance
    include("aggregates.jl")     # Aggregates    ← Parameters, Grids, Prices,
                                 #                  Household, Production
    include("distribution.jl")   # Distribution  ← Parameters, Grids,
                                 #                  Household, Inheritance
    include("equilibrium.jl")    # Equilibrium   ← all of the above
    include("reporting.jl")      # Reporting     ← Household, Inheritance, Equilibrium
    include("reform_budget.jl")  # ReformBudget  ← Inheritance, Equilibrium, Reporting
    include("calibration.jl")    # Calibration   ← Parameters, Grids, Equilibrium,
                                 #                  Reporting, Inheritance
end

# --- Bring exported names into scope ---
using .OLGModel.Parameters:  baseline_parameters
using .OLGModel.Grids:        build_grids
using .OLGModel.Equilibrium:  GESolverOptions
using .OLGModel.Inheritance:  BequestInheritanceParams
using .OLGModel.Reporting:    print_macro_summary, macro_stats,
                               wealth_gini, top_wealth_shares,
                               entrepreneur_top_stats, bequest_stats,
                               welfare_CE_by_age, welfare_CE_by_skillbins,
                               welfare_CE_by_wealthbins, welfare_CE_by_occupation
using .OLGModel.Calibration:  CalibrationTargets, CalibrationOptions, CalibrationResult,
                               calibrate, compute_moments,
                               save_calibration, load_calibration
using .OLGModel.ReformBudget: solve_reform_balanced_tau

using Printf

# ==============================================================================
# TeeIO — write to two streams simultaneously
# ==============================================================================

struct TeeIO <: IO
    primary   :: IO
    secondary :: IO
end
Base.write(t::TeeIO, x::UInt8)                    = (write(t.primary, x); write(t.secondary, x); 1)
Base.write(t::TeeIO, x::AbstractVector{UInt8})    = (write(t.primary, x); write(t.secondary, x); length(x))
Base.flush(t::TeeIO)                               = (flush(t.primary); flush(t.secondary); nothing)

# ==============================================================================
# Settings — edit these for each run
# ==============================================================================

# Path for saving / loading the calibrated baseline
const CALIBRATION_FILE = "baseline_calibration.jls"

# Set true to re-run calibration even when CALIBRATION_FILE already exists
const FORCE_RECALIBRATE = false

# Set true to re-run the initial GE solve even when its cache file already exists.
# The cache file is auto-keyed by (z_scale, F) so it is safe to leave this false;
# a new cache file is created automatically whenever you change those parameters.
const FORCE_REINIT_GE = false

# GE solver verbosity inside reform loops (set true if you need to debug)
const REFORM_VERBOSE = true

# Save all Step-3 results to a text file (Steps 1 & 2 always go to stdout only)
const SAVE_OUTPUT  = true
const RESULTS_FILE = "results.txt"

# ==============================================================================
# STEP 1 — Calibration
# ==============================================================================

println("\n", "="^62)
println("STEP 1 — CALIBRATION")
println("="^62)

cal = if !FORCE_RECALIBRATE && isfile(CALIBRATION_FILE)
    println("Saved calibration found — loading from: $CALIBRATION_FILE")
    load_calibration(CALIBRATION_FILE)
else
    println("No saved calibration found — running calibration now.")
    println("(Start Julia with  --threads auto  for parallel Jacobian columns.)")
    println()

    p0 = baseline_parameters()

    # Cache file is keyed by starting (z_scale, F) — safe to change parameters
    # freely without setting FORCE_REINIT_GE: a fresh cache is written automatically.
    _init_ge_file = @sprintf("init_ge_zs%.4f_F%.4f.jls", p0.z_scale, max(p0.F, 0.0))

    cal_result = @time calibrate(p0;
        targets = CalibrationTargets(
            ent_wealth_share = 0.416,
            ent_pop_share    = 0.115,
            debt_to_Y        = 1.52
        ),
        init_ge_file    = _init_ge_file,
        force_reinit_ge = FORCE_REINIT_GE,
        opts = CalibrationOptions(
            verbose           = true,
            parallel_jacobian = true,
            # Inner GE tolerances: looser than the defaults (1e-8) because
            # calibration only needs moments matched to 1e-4.  Cuts cost ~10–50x.
            ge_opts = GESolverOptions(
                verbose      = true,   # ← set false again after smoke test passes
                verbose_Ae   = false,
                # λ_Ae=0.05: small step avoids Ae–b_pension oscillation;
                # needs more iterations but converges monotonically.
                λ_Ae         = 0.05,
                λ_b          = 0.15,
                tol_Ae       = 5e-5,   # relaxed: calibration only needs 1e-4
                tol_w        = 5e-5,   # relaxed: w error of 3.4e-6 was causing spurious WARNINGs
                tol_K        = 1e-4,   # relaxed: moment error from K-gap of 1e-4 is ~1e-7; saves ~4 r-bisection steps
                tol_r        = 1e-5,   # relaxed: r accuracy of 1e-5 is sufficient for moments at 1e-4
                tol_DBN      = 1e-5,
                maxit_Ae     = 400,    # ~200 steps needed with λ_Ae=0.05
                max_iter_DBN = 200,    # increased from 150: safety margin for warm-start DBN convergence
                w_lo         = 0.05,   # need bracket below equilibrium wage
                w_hi         = 5.0,    # eq. wage is above 0.8; wImp(0.8)≈1.66 in diagnostic
                # r_init above r* (≈4.2% = 1/β-1) so foundA expands DOWN and finds
                # the bracket in 3-4 steps.  Without this, r0=geometric_mean(0.001,0.08)
                # =0.009 < r*, so foundA wastes 40 evaluations going the wrong direction.
                r_init       = 0.06
            )
        )
    )

    save_calibration(cal_result, CALIBRATION_FILE)
    cal_result
end

# --- Unpack calibrated economy ---
p_cal        = cal.res.p    # ModelParameters with calibrated z_scale, F, θ_vec
g_cal        = cal.res.g    # ModelGrids consistent with p_cal
res_baseline = cal.res      # Full EqmResults for the baseline steady state

println()
@printf("Calibration converged : %s\n", cal.converged)
@printf("Iterations used       : %d\n", cal.iter)
@printf("z_scale               : %.6f\n", cal.z_scale)
@printf("F (entry cost)        : %.6f\n", cal.F)
@printf("θ_scalar              : %.6f\n", cal.θ_scalar)

println()
print_macro_summary(res_baseline; title = "Baseline economy (calibrated)")

# ==============================================================================
# STEP 2 — Reforms
# ==============================================================================

println("\n", "="^62)
println("STEP 2 — REFORMS")
println("="^62)

# GE solver options shared by all reforms.
# Warm-starting from res_baseline is handled inside solve_reform_balanced_tau
# via the init_res keyword.
reform_ge_opts = GESolverOptions(
    verbose    = REFORM_VERBOSE,
    verbose_Ae = true
)

# ------------------------------------------------------------------------------
# Reform: balanced-budget inheritance tax financing a universal newborn transfer
#
# For each transfer level T, solve_reform_balanced_tau finds the inheritance
# tax rate τ_b that makes tax revenue equal to spending T × (mass of newborns).
# ------------------------------------------------------------------------------

transfer_levels = [20000.0, 50000.0]   # ← adjust to your thesis experiments

reform_results = Dict{Float64, NamedTuple}()

for T in transfer_levels
    println("\n--- Reform: newborn transfer T = $T ---")

    ref = solve_reform_balanced_tau(p_cal, g_cal;
        T         = T,
        exemption = 0.0,
        opts_ge   = reform_ge_opts,
        init_res  = res_baseline,      # warm-start from calibrated baseline
        verbose   = true
    )

    reform_results[T] = ref

    @printf("  Balanced-budget τ_b : %.6f\n", ref.τ)
    @printf("  Tax revenue         : %.4f\n", ref.taxrev)
    @printf("  Transfer spending   : %.4f\n", ref.spend)

    print_macro_summary(ref.res; title = "Reform economy (T=$T)")
end

# ==============================================================================
# STEP 3 — RESULTS & COMPARISON
# ==============================================================================

# Open the output file (or use a no-op sink when SAVE_OUTPUT is false)
_out_io = SAVE_OUTPUT ? open(RESULTS_FILE, "w") : devnull
_tee    = SAVE_OUTPUT ? TeeIO(stdout, _out_io)  : stdout

redirect_stdout(_tee) do

println("\n", "="^62)
println("STEP 3 — RESULTS & COMPARISON")
println("="^62)

# Collect all economies in order: baseline first, then each reform
all_labels = String["Baseline";  ["Reform T=$(Int(T))" for T in transfer_levels]]
all_res    =        [res_baseline; [reform_results[T].res for T in transfer_levels]]
all_τ_b    = Float64[0.0;          [reform_results[T].τ  for T in transfer_levels]]

# Pre-compute all reporting objects (one pass per economy)
all_ms   = [macro_stats(res)                               for res in all_res]
all_gini = [wealth_gini(res)                               for res in all_res]
all_top  = [top_wealth_shares(res; ps=[0.10, 0.01, 0.001]) for res in all_res]
all_ets  = [entrepreneur_top_stats(res; p=0.01)            for res in all_res]
all_bq   = [bequest_stats(res)                             for res in all_res]

# Printing helpers
function _hdr()
    @printf("%-32s", "")
    for lbl in all_labels;  @printf("  %18s", lbl);  end
    println()
end
function _row(label, vals)
    @printf("%-32s", label)
    for v in vals;  @printf("  %18s", v);  end
    println()
end

# ── (a) Macro aggregates ────────────────────────────────────────────────────────
println("\n─── (a) Macro aggregates ────────────────────────────────────────────")
_hdr()
_row("r  (interest rate)",     [@sprintf("%.6f",  ms.r)              for ms in all_ms])
_row("w  (wage)",              [@sprintf("%.4f",  ms.w)              for ms in all_ms])
_row("Y  (output)",            [@sprintf("%.4f",  ms.Y)              for ms in all_ms])
_row("K_supply / Y",           [@sprintf("%.4f",  ms.KY_supply)      for ms in all_ms])
_row("K_demand / Y",           [@sprintf("%.4f",  ms.KY_demand)      for ms in all_ms])
_row("Debt / Y",               [@sprintf("%.4f",  ms.Debt_to_Y)      for ms in all_ms])
_row("Ent share (pop)",        [@sprintf("%.4f",  ms.Ent_share_pop)  for ms in all_ms])
_row("Ent share (work-age)",   [@sprintf("%.4f",  ms.Ent_share_work) for ms in all_ms])
_row("TFP_Q",                  [@sprintf("%.6f",  ms.TFP_Q)          for ms in all_ms])
_row("rK mean",                [@sprintf("%.6f",  ms.rK_mean)        for ms in all_ms])

# ── (b) Wealth distribution ────────────────────────────────────────────────────
println("\n─── (b) Wealth distribution ─────────────────────────────────────────")
_hdr()
_row("Gini (wealth)",            [@sprintf("%.4f", g)                              for g   in all_gini])
_row("Top 10% wealth share",     [@sprintf("%.4f", t.top_100bp)                   for t   in all_top])
_row("Top  1% wealth share",     [@sprintf("%.4f", t.top_10bp)                    for t   in all_top])
_row("Top 0.1% wealth share",    [@sprintf("%.4f", t.top_1bp)                     for t   in all_top])
_row("Ent wealth share (pop)",   [@sprintf("%.4f", e.pop.overall_ent_wealth_share) for e   in all_ets])
_row("Ent wealth share top 1%",  [@sprintf("%.4f", e.pop.ent_wealth_share_top)    for e   in all_ets])
_row("Ent pop share  top 1%",    [@sprintf("%.4f", e.pop.ent_pop_share_top)       for e   in all_ets])

# ── (c) Bequest & reform statistics ────────────────────────────────────────────
println("\n─── (c) Bequest & reform statistics ─────────────────────────────────")
_hdr()
_row("τ_b (inheritance tax)",    [@sprintf("%.6f", τ)                              for τ   in all_τ_b])
_row("Bgross / K_supply",        [@sprintf("%.4f", bq.Bgross_over_K)              for bq  in all_bq])
_row("Tax revenue flow",         [@sprintf("%.4f", bq.TaxRev_flow)                for bq  in all_bq])
_row("Bequest median (gross)",   [@sprintf("%.2f", bq.percentiles_gross[0.50])    for bq  in all_bq])
_row("Bequest p90   (gross)",    [@sprintf("%.2f", bq.percentiles_gross[0.90])    for bq  in all_bq])
_row("Bequest p99   (gross)",    [@sprintf("%.2f", bq.percentiles_gross[0.99])    for bq  in all_bq])

# ── (d) Welfare CEV — each reform vs baseline ──────────────────────────────────
# Skill bins over the 9-point Guvenen z-grid: low / medium / high ability
skill_bins   = [[1,2,3], [4,5,6], [7,8,9]]
skill_labels = ["low  (z1–z3)", "medium (z4–z6)", "high (z7–z9)"]

# Wealth-bin labels matching the qs below (deciles + top 1%)
wealth_bin_labels = ["0–10%", "10–20%", "20–30%", "30–40%", "40–50%",
                     "50–60%", "60–70%", "70–80%", "80–90%", "90–99%", "top 1%"]

for T in transfer_levels
    res_ref = reform_results[T].res
    τ_b     = reform_results[T].τ

    println()
    println("─── (d) Welfare CEV: Reform T=$(Int(T))  " *
            "(τ_b=$(round(τ_b; digits=4)))  vs Baseline ───────────")

    # By occupation
    occ = welfare_CE_by_occupation(res_baseline, res_ref)
    println("  By occupation:")
    @printf("    %-20s  %+.4f %%\n", "workers",       occ.worker.CE       * 100)
    @printf("    %-20s  %+.4f %%\n", "entrepreneurs", occ.entrepreneur.CE * 100)

    # By wealth bin
    wb = welfare_CE_by_wealthbins(res_baseline, res_ref;
             qs = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50,
                   0.60, 0.70, 0.80, 0.90, 0.99, 1.0])
    println("  By wealth bin (baseline distribution weights):")
    for (b, lbl) in enumerate(wealth_bin_labels)
        @printf("    %-20s  %+.4f %%\n", lbl, wb.CE[b] * 100)
    end

    # By entrepreneurial ability bin
    sb = welfare_CE_by_skillbins(res_baseline, res_ref; bins = skill_bins)
    println("  By entrepreneurial ability:")
    for (b, lbl) in enumerate(skill_labels)
        @printf("    %-20s  %+.4f %%\n", lbl, sb.CE[b] * 100)
    end

    # By age — every 5th age plus the final age
    age_ce = welfare_CE_by_age(res_baseline, res_ref)
    println("  By age:")
    for h in vcat(collect(1:5:p_cal.MaxAge), [p_cal.MaxAge])
        h <= length(age_ce.CE) || continue
        isnan(age_ce.CE[h])    && continue
        status = h < p_cal.RetAge ? "working" : "retired"
        @printf("    age %2d (%s)  %+.4f %%\n", h, status, age_ce.CE[h] * 100)
    end
end

# ==============================================================================
# Done
# ==============================================================================

println("\n", "="^62)
println("Run complete.")
println("="^62)

end  # redirect_stdout

SAVE_OUTPUT && close(_out_io)
SAVE_OUTPUT && println("\nResults saved to: $RESULTS_FILE")