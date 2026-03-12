# OLG Model — Master's Thesis: Efficiency Effects of a Universal Basic Capital Financed by an Inheritance Tax 
Author: Loris Sultano
Contact: loris_sultano@mailbox.org

Julia implementation of an Overlapping Generations model with endogenous entrepreneurship,
dynastic ability inheritance, warm-glow bequests, and a balanced-budget inheritance tax reform.
The model closely follows Guvenen et al. (2023) and is calibrated to U.S. data.

---

## Quick Start

### Requirements

- Julia ≥ 1.9 (tested on 1.10)
- No external packages — all modules use Julia's standard library only
  (`LinearAlgebra`, `Printf`, `Serialization`)

### Running the model

```
julia --threads auto main.jl
```

`--threads auto` enables parallel Jacobian evaluation during calibration (~3× speedup)
and parallel backward induction in the VFI. On a single-core machine the code still runs correctly.

Output is written to stdout. Set `SAVE_OUTPUT = true` in [main.jl](main.jl) to also write
Step 3 results to `results.txt`.

---

## File Overview

```
main.jl            Entry point — loads all modules, runs calibration and reforms
parameters.jl      Structural parameters (ModelParameters struct + baseline_parameters())
demographics.jl    Bell–Miller survival probabilities
prices.jl          ModelPrices struct (r, w)
grids.jl           Asset grid + entrepreneurial ability grid construction
utility.jl         CRRA utility u(c) and warm-glow bequest utility v(b)
production.jl      Final-good and entrepreneurial technology blocks
inheritance.jl     Dynastic skill inheritance (Markov transition matrix) + bequest tax rules
household.jl       Value function iteration (backward induction) with occupational choice
aggregates.jl      Aggregate objects from the stationary distribution
distribution.jl    Stationary distribution iteration (forward induction)
equilibrium.jl     GE solver: nested fixed-point for (r, w, A_e, b_pension)
calibration.jl     Broyden calibration to three empirical moments
reporting.jl       Macro statistics, inequality measures, welfare CEV
reform_budget.jl   Balanced-budget inheritance tax reform solver
```

---

## Model Structure

### Demographics

**81 model periods** (ages 20–100). Working ages: periods 1–44; retirement: periods 45–81.
Population weights and conditional survival probabilities come from Bell–Miller (2002),
hard-coded in `demographics.jl` to match the original Fortran implementation.

### Preferences

Agents maximise expected lifetime utility

```
E Σ_{h=1}^{H} β^{h-1} · (Π_{j=1}^{h-1} s_j) · u(c_h)  +  (1 - s_h) · v(b_{h+1})
```

- **CRRA period utility**: `u(c) = c^(1-σ) / (1-σ)`, with σ = 4.
- **Warm-glow bequest utility**: `v(b) = χ_bq · (b + bq_0)^(1-σ) / (1-σ)`.
- Discount factor β = 0.9593.

### Technology

Two-sector structure:

1. **Final good**: Cobb–Douglas, `Y = Q^α · L^(1-α)`, α = 0.40.
   - Competitively produced; labour L is hired from workers; Q is a CES aggregate.
   - Factor prices: `w = (1-α) Y / L`, `pQ = α Y / Q`.

2. **Intermediate good** (entrepreneur aggregation):
   - Each working-age entrepreneur with ability z and capital k produces `(z·k)^μ` units.
   - CES aggregate: `Q = (Σ_i (z_i k_i)^μ · mass_i)^(1/μ)`, μ = 0.90.
   - Entrepreneurial productivity scale `A_e` is endogenous (see GE solver).

### Entrepreneurial ability and financial constraint

- Ability z drawn from a 9-point discretization of the Guvenen et al. grid, rescaled by `z_scale`.
- z is **fixed within life** and transmitted dynastically via an AR(1) in logs:
  `log(z_child) = ρ_z · log(z_parent) + ε`, ε ~ N(0, σ²_ε).
- Borrowing constraint: `k ≤ θ(z) · a`.
  - `θ_vec` is a 9-element vector that increases with ability rank.
  - All elements are scaled by a single `θ_scalar` (calibrated jointly).

### Occupational choice

Each working-age agent chooses `e ∈ {0, 1}`:
- `e = 0`: worker, earns wage `(1 - τ_pension) · w`; no capital required.
- `e = 1`: entrepreneur, chooses capital `k ≤ θ(z) · a`, earns profit
  `π = A_e (z k)^μ − (r + δ) k`.

Switching from worker to entrepreneur requires paying a one-time entry cost F (calibrated).
The previous-period occupation `ep ∈ {1, 2}` is tracked as a state variable to apply F correctly.

### Pension system

PAYG (pay-as-you-go):
- Fixed contribution rate τ_pension = 0.124 applies to wages and positive entrepreneurial profits.
- Benefit `b_pension` is set so that revenue equals expenditure each period (balanced PAYG).
- `b_pension` is part of the inner GE fixed point.

### Bequests and inheritance tax

- Dying agents leave assets `a_next` as gross bequests.
- Net bequest: `b_net = max(0, b_gross − τ_b · max(0, b_gross − exemption)) + transfer`.
- Newborns inherit ability from their parent via the dynastic Markov matrix and start life
  with assets equal to `b_net` mapped to the nearest grid point.

---

## Module Dependency Graph

```
Demographics
    └─► Parameters ──────────────────────────────────────────┐
             └─► Grids                                        │
             └─► Prices                                       │
             └─► Utility ◄────────────────────────────────── │
             └─► Production ◄──── Parameters, Prices, Utility │
             └─► Inheritance ◄─── Parameters, Grids          │
             └─► Household ◄───── Parameters, Grids, Prices, │
             │                    Utility, Production,        │
             │                    Inheritance                 │
             └─► Aggregates ◄──── Parameters, Grids, Prices, │
             │                    Household, Production       │
             └─► Distribution ◄── Parameters, Grids,         │
             │                    Household, Inheritance      │
             └─► Equilibrium ◄─── all of the above           │
             └─► Reporting ◄───── Household, Inheritance,    │
             │                    Equilibrium                 │
             └─► ReformBudget ◄── Inheritance, Equilibrium,  │
             │                    Reporting                   │
             └─► Calibration ◄─── Parameters, Grids,         │
                                  Equilibrium, Reporting,    │
                                  Inheritance                 │
```

All modules are wrapped inside a single parent module `OLGModel` in `main.jl`,
so `import ..Parameters` inside any submodule refers to the sibling module.

---

## Module Reference

### `demographics.jl` — Demographics

Provides the Bell–Miller (2002) age-81 population vector and computes conditional
survival probabilities `survP[h] = pop[h+1] / pop[h]` (survP[81] = 0).

Key functions:
- `bell_miller_pop_surv()` → `(pop, survP)` — used once in `baseline_parameters()`.

---

### `parameters.jl` — Parameters

Defines `ModelParameters` (immutable struct with `@kwdef`) and `baseline_parameters()`.

Key parameters:

| Symbol      | Value   | Meaning |
|-------------|---------|---------|
| MaxAge      | 81      | model periods (ages 20–100) |
| RetAge      | 45      | first retirement period |
| β           | 0.9593  | annual discount factor |
| σ           | 4.0     | CRRA coefficient |
| χ_bq        | 0.2     | bequest utility weight |
| bq_0        | 26 800  | bequest shifter (floor) |
| α           | 0.40    | capital share in final good |
| δ           | 0.05    | annual depreciation rate |
| μ           | 0.90    | returns-to-scale in entrepreneur technology |
| τ_pension   | 0.124   | PAYG contribution rate |
| nz          | 9       | grid points for entrepreneurial ability |
| ρz          | 0.1     | intergenerational ability persistence |
| σz_eps      | 0.277   | innovation s.d. for dynastic AR(1) |
| na          | 100     | asset grid points (set to 51 for testing) |
| amax        | 500 000 | maximum assets |
| a_theta     | 4.0     | asset grid curvature (higher = denser near amin) |

`with(p; kwargs...)` returns a modified copy — used throughout the solver to avoid
repeating the full constructor when changing one or two fields.

---

### `prices.jl` — Prices

A thin container:
```julia
struct ModelPrices
    r::Float64   # interest rate
    w::Float64   # wage
end
```
Passed to every function that needs factor prices.

---

### `grids.jl` — Grids

Builds the two discretized state spaces:

- **Asset grid**: convex grid on `[amin, amax]` using power-law spacing
  `a_i = (i * amax^(1/θ) / na)^θ`. Higher `a_theta` concentrates points near zero.
- **Ability grid**: the 9-point Guvenen et al. grid, rescaled by `z_scale` (calibrated).
  Stationary mass `Gz` is hard-coded to match the original Fortran code.

`nearest_index(xgrid, x)` maps a continuous value to the closest grid index.

---

### `utility.jl` — Utility

```julia
u(c, p)         # CRRA:        c^(1-σ) / (1-σ),  returns BIG_NEG if c ≤ 0
v_bequest(b, p) # warm-glow:   χ_bq · (b + bq_0)^(1-σ) / (1-σ), returns BIG_NEG if b < 0
```

Both use `c^Int(1-σ)` (integer exponent dispatch) to avoid the ~10–20× slower
`exp(log(c) * (1-σ))` float-power code path.

---

### `production.jl` — Production

Implements both production blocks.

**Final-good block** (competitive):
- `final_good_block(Q, L, α)` → `FinalGoodResults(Y, w, pQ)`.
- Marginals: `w = (1-α) Q^α L^(-α)`, `pQ = α Q^(α-1) L^(1-α)`.

**Entrepreneur block** (monopolistic):
- `k_star(z, p, prices)` — unconstrained optimal capital `k* = (μ A_e z^μ / (r+δ))^(1/(1-μ))`.
- `entrepreneur_profit_k(a, z, iz, p, prices)` — constrained profit and capital,
  applying the borrowing constraint `k ≤ θ(iz) · a`.
- `implied_Ae(pQ, Q, μ)` — backs out the productivity scale `A_e = pQ · Q^(1-μ)` consistent
  with competitive intermediate-good pricing.

---

### `inheritance.jl` — Inheritance

Two components:

1. **Skill inheritance** (dynastic AR(1)):
   - `skill_transition_matrix(zgrid, shp)` — builds the nz × nz Markov matrix `Pz`
     using log-midpoint cutoffs and a self-contained Abramowitz–Stegun Normal CDF.
   - `stationary_skill_dist(zgrid, p)` — returns the ergodic distribution `Gz` over `zgrid`
     (used to initialize `DBN` and to seed the ability grid).

2. **Bequest tax**:
   - `BequestInheritanceParams(τ_b, exemption, transfer)` — policy parameters.
   - `net_bequest(b_gross, bip)` — applies the tax:
     `b_net = max(0, b_gross − τ_b · max(0, b_gross − exemption)) + transfer`.

---

### `household.jl` — Household

Solves the household problem by **backward induction** over ages h = MaxAge, …, 1.

State space: `(h, ia, iz, ep)` where ep ∈ {1, 2} encodes the previous-period occupation
(1 = worker, 2 = entrepreneur; relevant for charging the entry cost F when switching).

At each state node:
1. Evaluate worker utility: consume `c = (1+r)a + y_w - a_next` for each `a_next`.
2. Evaluate entrepreneur utility: consume `c = (1+r)a + π(a,z) - a_next` for each `a_next`.
3. Choose `e* = argmax {V_worker, V_entrepreneur}`.
4. Apply warm-glow bequest to the terminal-age Bellman.
5. Survival probability weights the continuation value; death triggers bequest utility.

Returns `Policies(c, ia_next, e, k)` — all policy arrays of size `(MaxAge, na, nz, 2)`.

The inner grid search (optimal savings) uses golden-section or grid-search over `ia_next`.
Parallelized over the ability index with `@threads`.

---

### `aggregates.jl` — Aggregates

Given the stationary distribution `DBN[h, ia, iz, ep]` and policy functions, computes:

| Object | Definition |
|--------|-----------|
| K_supply | ∫ a dμ (household assets = total savings) |
| K_demand | ∫ k dμ (capital used by entrepreneurs) |
| Debt | ∫ max(k−a, 0) dμ (external business financing) |
| L | ∫ 1(e=0) dμ (mass of workers, working ages only) |
| Q | (∫ (zk)^μ dμ)^(1/μ) (CES intermediate aggregate) |
| b_pension_implied | τ_pension · tax_base / RetPop (PAYG balance) |
| Ent_share_work | entrepreneurs / working-age population |
| Ent_share_pop | entrepreneurs / total population |

Only working-age agents (h < RetAge) contribute to L, K_demand, and the pension tax base.

---

### `distribution.jl` — Distribution

Computes the **stationary distribution** by iterating the forward Kolmogorov equation until
`max |DBN_new − DBN| < tol_DBN`.

Transition rules per period:
- **Survivors**: mass at `(h, ia, iz, ep)` moves to `(h+1, ia_next[h,ia,iz,ep], iz, e_chosen+1)`.
- **Deaths**: dying mass is redistributed to newborns `(h=1, ia_bequest, iz_child, ep=1)` using
  the dynastic transition matrix `Pz` for the child's ability draw and `bip` for the net bequest.
- Newborns start with `ep = 1` (no prior occupation), so they pay entry cost F if they choose
  entrepreneurship in period 1.

Optional damping (`α` parameter) accelerates convergence for dynasty problems.

---

### `equilibrium.jl` — Equilibrium

The GE solver has **four nested layers**:

```
Layer 1 (outermost): r-bisection
  │  finds r* such that K_supply(r*) - K_demand(r*) = 0
  │
  └─► Layer 2: w fixed-point (damped log-space FP)
        │  finds w* such that w_implied(w*) = w*
        │
        └─► Layer 3: (A_e, b_pension) fixed-point
              │  given prices (r, w): iterates A_e and b_pension to
              │  consistency with the production block and PAYG balance
              │
              └─► Layer 4 (innermost): VFI + Distribution
                    solve_household(p, g, prices)
                    stationary_distribution(p, g, pol)
                    compute_aggregates(p, g, pol, DBN, prices)
```

**Layer 4** is the computational bottleneck. Each evaluation requires a full VFI backward pass
(O(na² × nz × MaxAge)) plus a distribution forward pass (O(na × nz × MaxAge)).

**Layer 3** (`solve_Ae_fixedpoint_given_prices`): damped fixed point,
`A_e ← (1−λ_Ae) A_e + λ_Ae · implied_Ae(...)`, with λ_Ae = 0.05.
Also jointly updates `b_pension`. Converges in ~100–400 iterations with small λ_Ae.

**Layer 2** (`solve_w_Ae_given_r`): damped log-space fixed point,
`log w ← log w + λ_w_inner · (log w_implied − log w)`, with λ_w_inner = 0.5.
Warm-starts Layer 3 from the previous w iterate (consecutive w values are close, so Layer 3
re-converges in ~5–10 steps instead of ~100). Converges in ~10–20 w iterations.

**Layer 1** (`solve_GE`): bisection on r in `[r_lo, r_hi]`.
Set `r_init = 0.06` to start above r* ≈ 1/β−1 ≈ 4.2%, so the bracket-finding step
expands in the correct direction and wastes no evaluations.

**Key solver options** (`GESolverOptions`):

| Option | Default | Meaning |
|--------|---------|---------|
| `r_init` | NaN | starting r; set to 0.06 for this calibration |
| `w_lo`, `w_hi` | 0.1, 50 | wage bracket; use `w_lo=0.05, w_hi=5.0` |
| `λ_Ae` | 0.3 | Ae damping; use 0.05 to avoid oscillation |
| `maxit_Ae` | 200 | Ae max iterations; use 400 with λ_Ae=0.05 |
| `λ_w_inner` | 0.5 | w FP step size (log-space) |
| `tol_Ae` | 1e-8 | Ae convergence tolerance |
| `tol_DBN` | 1e-8 | DBN convergence tolerance |
| `accept_w_relerr` | 0.10 | tolerance for w non-convergence before error |

---

### `calibration.jl` — Calibration

Jointly calibrates three structural parameters to three empirical moments
using **Broyden's quasi-Newton method**:

| Parameter | Moment targeted |
|-----------|----------------|
| `z_scale` | entrepreneur share of total population wealth (target: 41.6%) |
| `F` | entrepreneur share of total population (target: 11.5%) |
| `θ_scalar` | business debt / output ratio (target: 1.52) |

`θ_scalar` multiplies the full `θ_vec` component-wise (preserves the ability-dependent shape).

Algorithm:
1. Cold solve at the initial point.
2. Estimate the 3×3 Jacobian by forward finite differences (3 GE solves, run in parallel
   when `--threads auto` is used and `parallel_jacobian=true`).
3. Broyden rank-1 updates with backtracking line search until `||r||∞ < tol = 1e-4`.

The calibrated result is serialized to `baseline_calibration.jls` via Julia's built-in
`Serialization` module. On subsequent runs, `main.jl` loads from disk and skips the solve
(controlled by `FORCE_RECALIBRATE`).

---

### `reporting.jl` — Reporting

Post-solution statistics extracted from an `EqmResults` object:

| Function | Output |
|----------|--------|
| `macro_stats(res)` | r, w, Y, C, K/Y, debt/Y, entrepreneur shares, TFP |
| `print_macro_summary(res)` | formatted console print of macro stats |
| `life_cycle_profiles(res)` | mean assets, consumption, entrepreneur share by age |
| `wealth_gini(res)` | Gini coefficient of asset distribution |
| `top_wealth_shares(res)` | top-1%, top-0.1% wealth shares |
| `entrepreneur_top_stats(res)` | entrepreneur over-representation in top wealth decile |
| `bequest_stats(res)` | bequest flows, tax revenue, bequest percentiles |
| `welfare_CE_by_age(res0, res1)` | CEV by age (consumption-equivalent variation) |
| `welfare_CE_by_skillbins(res0, res1)` | CEV by ability quartile |
| `welfare_CE_by_wealthbins(res0, res1)` | CEV by wealth quartile |
| `welfare_CE_by_occupation(res0, res1)` | CEV for workers vs entrepreneurs |

CEV formula: `CE = (V1/V0)^{1/(1-σ)} − 1`, where V0, V1 are weighted-average values
under baseline and reform, weighted by the baseline (or reform) DBN.

---

### `reform_budget.jl` — ReformBudget

Solves a **balanced-budget inheritance tax reform**: find `τ_b` such that inheritance
tax revenue exactly covers a flat lump-sum transfer T to each newborn.

Budget constraint: `τ_b · ∫ max(0, b − exemption) · death_prob dμ = T · deaths_mass`.

`solve_reform_balanced_tau(p, g; T, exemption, ...)`:
1. Evaluate residual at `τ_lo = 0` and `τ_hi = 0.6`.
2. Auto-expand `τ_hi` if both endpoints have the same sign.
3. Bisect to `tol_rel = 1e-6` relative gap.
4. Each GE solve warm-starts from the previous `r` and `DBN`.

---

## Calibration Targets and Data Sources

| Moment | Target | Source |
|--------|--------|--------|
| Entrepreneur wealth share | 41.6% | Quadrini (2000), SCF |
| Entrepreneur population share | 11.5% | SCF / CPS |
| Business debt / Y | 1.52 | Flow of Funds |

Demographics: Bell–Miller (2002), as used in Guvenen et al. (2023).

---

## Output Files

| File | Contents |
|------|----------|
| `baseline_calibration.jls` | serialized `CalibrationResult` (created on first run) |
| `results.txt` | Step 3 reform output (created if `SAVE_OUTPUT = true` in main.jl) |

---

## Runtime Notes

All timings are for a 2019 MacBook Air (thermal throttling to ~1.6 GHz after ~30 minutes).

| Grid | Estimated runtime |
|------|------------------|
| na = 51 | ~1–3 days |
| na = 100 | ~5–14 days |
| na = 201 | not feasible on laptop |

Key runtime-reducing decisions made during development:
- `c^Int(1-σ)` instead of `c^(1-σ::Float64)` → ~10-20× VFI speedup (integer power dispatch).
- `r_init = 0.06` → avoids ~40 wasted r evaluations when r* > r0.
- w fixed-point (FP) instead of w bisection → ~20–30× inner-loop speedup via warm-starting.

---

---

## Adding Progress Bars Only (Without Performance Improvements)

If you only want to add the progress bar functionality without the parallelization improvements, follow these steps:

### Files to Keep As-Is (Progress Bars Only)

These files contain only progress bar changes and can be kept entirely:

1. **`progress.jl`** — New file, keep entirely
2. **`main.jl`** — Only change: added `include("progress.jl")` line (around line 22)

### Files with Mixed Changes (Keep Progress Bars, Revert Parallelization)

These files have both progress bars and parallelization. You need to keep the progress bar parts but revert the parallelization changes:

#### `calibration.jl`

**Keep:**
- Import: `import ..Progress` (around line 16)
- Progress bar creation: `pb_jac = opts.verbose ? Progress.ProgressBar(...)` (around line 278)
- Progress bar updates: `opts.verbose && Progress.update!(pb_jac, j)` (around lines 287, 291)
- Progress bar finish: `opts.verbose && Progress.finish!(pb_jac)` (around line 295)
- Progress bar creation: `pb_cal = opts.verbose ? Progress.ProgressBar(...)` (around line 302)
- Progress bar updates: `opts.verbose && Progress.update!(pb_cal, it)` (around line 305)
- Progress bar finish calls: `opts.verbose && Progress.finish!(pb_cal, ...)` (around lines 361, 367)

**Revert:**
- Remove: `using Base.Threads` (line 7)
- In `_ent_wealth_share_pop` function (around line 388):
  - Change back from parallelized version to sequential:
  ```julia
  # REVERT TO:
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
  ```

#### `equilibrium.jl`

**Keep:**
- Import: `import ..Progress` (around line 22)
- All progress bar creation, update, and finish calls for `pb_ae`, `pb_w`, and `pb_r`

**Revert:**
- No parallelization changes in this file (it only has progress bars)

#### `distribution.jl`

**Keep:**
- Import: `import ..Progress` (around line 11)
- Progress bar creation: `pb_dbn = Progress.ProgressBar(...)` (around line 98)
- Progress bar updates: `Progress.update!(pb_dbn, iter)` (around line 101)
- Progress bar finish calls: `Progress.finish!(pb_dbn, ...)` (around lines 164, 172)

**Revert:**
- Change parallelization back from `ia` to `iz`:
  - Line ~109: Change `Threads.@threads for ia in 1:na` back to `Threads.@threads for iz in 1:nz`
  - Swap the loop order: `for ia in 1:na` becomes inner loop, `for iz in 1:nz` becomes outer loop
  - Original structure was:
  ```julia
  Threads.@threads for iz in 1:nz
      @inbounds for h in 1:H
          for ia in 1:na
              for ep in 1:2
                  # ... code ...
              end
          end
      end
  end
  ```

### Files to Revert Entirely (Parallelization Only)

These files only have parallelization changes and should be reverted:

#### `household.jl`

**Revert:**
- Remove: `using Base.Threads` (line 3)
- Change parallelization back from `ia` to `iz`:
  - Around line 131: Change `Threads.@threads for ia in 1:na` back to `Threads.@threads for iz in 1:nz`
  - Around line 204: Same change
  - Swap loop order: `for iz in 1:nz` becomes outer parallel loop, `for ia in 1:na` becomes inner sequential loop
  - Original structure was:
  ```julia
  Threads.@threads for iz in 1:nz
      @inbounds begin
          z = g.zgrid[iz]
          for ia in 1:na
              # ... code ...
          end
      end
  end
  ```

#### `aggregates.jl`

**Revert:**
- Remove: `using Base.Threads` (line 5)
- Remove all thread-local accumulator arrays and parallelization
- Revert to sequential loop:
  ```julia
  # REVERT TO:
  @inbounds for h in 1:H
      working = (h < p.RetAge)
      for ia in 1:na
          a = g.agrid[ia]
          for iz in 1:nz
              for ep in 1:ne
                  # ... original sequential code ...
              end
          end
      end
  end
  ```

#### `reporting.jl`

**Revert:**
- Remove: `using Base.Threads` (line 10)
- In `aggregate_consumption` function (around line 57):
  - Revert to sequential version:
  ```julia
  # REVERT TO:
  function aggregate_consumption(DBN::Array{Float64,4}, pol::Household.Policies)
      H, na, nz, ne = size(DBN)
      C = 0.0
      @inbounds for h in 1:H, ia in 1:na, iz in 1:nz, ep in 1:ne
          m = DBN[h, ia, iz, ep]
          m == 0.0 && continue
          C += pol.c[h, ia, iz, ep] * m
      end
      return C
  end
  ```

### Quick Summary

**To add progress bars only:**
1. Keep `progress.jl` and the `include("progress.jl")` in `main.jl`
2. Keep all `Progress.ProgressBar`, `Progress.update!`, and `Progress.finish!` calls
3. Revert all `Threads.@threads` changes back to original parallelization strategy (over `iz` instead of `ia` or `h`)
4. Remove `using Base.Threads` from files that don't need it
5. Revert thread-local accumulator patterns back to simple sequential loops

The progress bars will work identically whether or not you keep the parallelization improvements.

---

## References

- Guvenen, F., G. Kambourov, B. Kuruscu, S. Ocampo, and D. Chen (2023). Use it or lose it: Efficiency and redistributional effects of wealth taxation. The Quarterly Journal of
Economics 138(2), 835–894.
- Bell, F.C., Miller, M.L. (2002). *Life tables for the United States Social Security Area 1900–2100*.
  Actuarial Study No. 116, SSA.
