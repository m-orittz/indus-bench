module Reporting

export macro_stats, print_macro_summary,
       life_cycle_profiles,
       wealth_gini, top_wealth_shares, entrepreneur_top_stats,
       bequest_stats,
       welfare_CE_by_age, welfare_CE_by_skillbins,
       welfare_CE_by_wealthbins, welfare_CE_by_occupation

using Base.Threads
using Printf

import ..Household
import ..Inheritance
using ..Equilibrium: EqmResults

"""
    _safe_div(num, den)

Return `num/den` if `den>0`, otherwise `NaN`.
"""
@inline function _safe_div(num::Float64, den::Float64)
    return den > 0.0 ? num / den : NaN
end

"""
    weighted_quantile(x, w, q)

Compute the weighted quantile of `x` with weights `w` at `q ∈ [0,1]`.

Algorithm:
- Sort `x` (and reorder weights accordingly)
- Return the smallest `x` such that the weighted CDF is at least `q`
"""
function weighted_quantile(x::AbstractVector{<:Real}, w::AbstractVector{<:Real}, q::Float64)
    @assert 0.0 <= q <= 1.0
    @assert length(x) == length(w)

    s = sum(w)
    s <= 0 && return NaN

    idx = sortperm(x)
    xS = Float64.(x[idx])
    wS = Float64.(w[idx])

    cw = cumsum(wS) ./ s
    j = searchsortedfirst(cw, q)
    j = clamp(j, 1, length(xS))
    return xS[j]
end

"""
    aggregate_consumption(DBN, pol)

Compute aggregate consumption `C = ∫ c dμ` using the stationary distribution `DBN`
and consumption policy `pol.c`.
"""
function aggregate_consumption(DBN::Array{Float64,4}, pol::Household.Policies)
    H, na, nz, ne = size(DBN)
    # Parallelize over age h (81 ages) for better thread utilization
    nt = Threads.nthreads()
    local_C = zeros(Float64, nt)
    @inbounds Threads.@threads for h in 1:H
        tid = Threads.threadid()
        for ia in 1:na, iz in 1:nz, ep in 1:ne
            m = DBN[h, ia, iz, ep]
            m == 0.0 && continue
            local_C[tid] += pol.c[h, ia, iz, ep] * m
        end
    end
    return sum(local_C)
end

"""
    macro_stats(res)

Return a NamedTuple with commonly used macro objects for tables and calibration.

This function relies on `res.agg` and `res.prod` computed by the GE solver.

It returns working-age and population versions of labor and entrepreneur shares.
For backward compatibility, it falls back to older field names if needed.
"""
function macro_stats(res::EqmResults)
    agg  = res.agg
    prod = res.prod

    Pop = agg.Pop
    Y   = prod.Y
    C   = aggregate_consumption(res.DBN, res.pol)

    has_work_fields = hasproperty(agg, :Ent_share_work) && hasproperty(agg, :L_share_work)

    Ent_share_work = has_work_fields ? agg.Ent_share_work : getproperty(agg, :Ent_share)
    L_share_work   = has_work_fields ? agg.L_share_work   : getproperty(agg, :L_share)

    Ent_share_pop  = has_work_fields ? agg.Ent_share_pop  : getproperty(agg, :Ent_share)
    L_share_pop    = has_work_fields ? agg.L_share_pop    : getproperty(agg, :L_share)

    WorkPop = hasproperty(agg, :WorkPop) ? agg.WorkPop : NaN

    return (
        r   = res.prices.r,
        w   = res.prices.w,
        Ae  = res.p.A_e,

        Pop = Pop,
        WorkPop = WorkPop,

        Y   = Y,
        C   = C,
        Y_per_cap = _safe_div(Y, Pop),
        C_per_cap = _safe_div(C, Pop),

        K_supply = agg.K_supply,
        K_demand = agg.K_demand,
        K_supply_per_cap = agg.K_supply_per_cap,
        K_demand_per_cap = agg.K_demand_per_cap,

        KY_supply = _safe_div(agg.K_supply, Y),
        KY_demand = _safe_div(agg.K_demand, Y),

        Debt = agg.Debt,
        Debt_to_Y = _safe_div(agg.Debt, Y),

        L = agg.L,

        L_share   = L_share_work,
        Ent_share = Ent_share_work,

        L_share_work   = L_share_work,
        Ent_share_work = Ent_share_work,
        L_share_pop    = L_share_pop,
        Ent_share_pop  = Ent_share_pop,

        Q = agg.Q,
        TFP_Q = agg.TFP_Q,
        TFP_Q_assets = agg.TFP_Q_assets,
        rK_mean = agg.rK_mean,

        pQ = prod.pQ
    )
end

"""
    print_macro_summary(res; title="Macro summary")

Print a compact macro summary to stdout and return `macro_stats(res)`.
"""
function print_macro_summary(res::EqmResults; title::String = "Macro summary")
    ms = macro_stats(res)

    println("\n=== ", title, " ===")
    @printf("r = %.6f   w = %.6f   A_e = %.6e\n", ms.r, ms.w, ms.Ae)
    @printf("Y = %.6f   C = %.6f   Pop = %.6f", ms.Y, ms.C, ms.Pop)

    if isfinite(ms.WorkPop)
        @printf("   WorkPop = %.6f\n", ms.WorkPop)
    else
        print("\n")
    end

    @printf("K_supply/Y = %.4f   K_demand/Y = %.4f   Debt/Y = %.4f\n", ms.KY_supply, ms.KY_demand, ms.Debt_to_Y)

    @printf("Ent_share_work = %.4f   L_share_work = %.4f\n", ms.Ent_share_work, ms.L_share_work)
    @printf("Ent_share_pop  = %.4f   L_share_pop  = %.4f\n", ms.Ent_share_pop,  ms.L_share_pop)

    @printf("Q = %.6f   TFP_Q = %.6f   TFP_Q_assets = %.6f   rK_mean = %.6f\n",
            ms.Q, ms.TFP_Q, ms.TFP_Q_assets, ms.rK_mean)

    return ms
end

"""
    life_cycle_profiles(res)

Return life-cycle profiles by age `h`:

- `mass[h]`
- `mean_assets[h]`
- `mean_consumption[h]`
- `entrepreneur_share[h]`
- `mean_k_among_entrepreneurs[h]`
"""
function life_cycle_profiles(res::EqmResults)
    DBN = res.DBN
    pol = res.pol
    g   = res.g
    H, na, nz, ne = size(DBN)

    mass       = zeros(Float64, H)
    mean_a     = fill(NaN, H)
    mean_c     = fill(NaN, H)
    ent_share  = fill(NaN, H)
    mean_k_ent = fill(NaN, H)

    @inbounds for h in 1:H
        m_h      = 0.0
        a_sum    = 0.0
        c_sum    = 0.0
        ent_m    = 0.0
        k_sum_ent = 0.0

        for ia in 1:na
            a = g.agrid[ia]
            for iz in 1:nz
                for ep in 1:ne
                    m = DBN[h, ia, iz, ep]
                    m == 0.0 && continue

                    m_h   += m
                    a_sum += a * m
                    c_sum += pol.c[h, ia, iz, ep] * m

                    if pol.e[h, ia, iz, ep] == 1
                        ent_m     += m
                        k_sum_ent += pol.k[h, ia, iz, ep] * m
                    end
                end
            end
        end

        mass[h]       = m_h
        mean_a[h]     = _safe_div(a_sum, m_h)
        mean_c[h]     = _safe_div(c_sum, m_h)
        ent_share[h]  = _safe_div(ent_m, m_h)
        mean_k_ent[h] = ent_m > 0 ? (k_sum_ent / ent_m) : 0.0
    end

    return (
        mass = mass,
        mean_assets = mean_a,
        mean_consumption = mean_c,
        entrepreneur_share = ent_share,
        mean_k_among_entrepreneurs = mean_k_ent
    )
end

"""
    wealth_gini(res)

Compute the Gini coefficient of the cross-sectional distribution of assets `a`
(using the stationary DBN aggregated over ages and `z`).
"""
function wealth_gini(res::EqmResults)
    DBN = res.DBN
    g   = res.g
    H, na, nz, ne = size(DBN)

    mass_a = zeros(Float64, na)
    @inbounds for ia in 1:na
        s = 0.0
        for h in 1:H, iz in 1:nz, ep in 1:ne
            s += DBN[h, ia, iz, ep]
        end
        mass_a[ia] = s
    end

    W = sum(mass_a)
    W <= 0 && return NaN

    X = sum(g.agrid .* mass_a)
    X <= 0 && return NaN

    cw = cumsum(mass_a) ./ W
    cx = cumsum(g.agrid .* mass_a) ./ X

    area = 0.0
    prev_cw = 0.0
    prev_cx = 0.0
    @inbounds for i in 1:na
        area += 0.5 * (cx[i] + prev_cx) * (cw[i] - prev_cw)
        prev_cw = cw[i]
        prev_cx = cx[i]
    end

    return 1.0 - 2.0 * area
end

"""
    top_wealth_shares(res; ps=[0.10, 0.01, 0.001])

Return wealth shares held by the top `p` fraction of the population, using assets `a`.

Implementation detail:
- Uses the distribution over the asset grid implied by DBN.
- Uses a partial-mass correction at the cutoff grid point to avoid degeneracy on coarse grids.
"""
function top_wealth_shares(res::EqmResults; ps = [0.10, 0.01, 0.001])
    DBN = res.DBN
    g   = res.g
    H, na, nz, ne = size(DBN)

    mass_a = zeros(Float64, na)
    @inbounds for ia in 1:na
        s = 0.0
        for h in 1:H, iz in 1:nz, ep in 1:ne
            s += DBN[h, ia, iz, ep]
        end
        mass_a[ia] = s
    end

    W = sum(mass_a)
    X = sum(g.agrid .* mass_a)
    (W <= 0 || X <= 0) && return NamedTuple()

    out = Dict{Float64, Float64}()
    for p in ps
        target = p * W
        m = 0.0
        wsum = 0.0

        @inbounds for ia in na:-1:1
            m_next = m + mass_a[ia]
            if m_next >= target
                need = target - m
                wsum += g.agrid[ia] * need
                m = target
                break
            else
                m = m_next
                wsum += g.agrid[ia] * mass_a[ia]
            end
        end

        out[p] = wsum / X
    end

    return (; (Symbol("top_$(Int(round(p*1000)))bp") => out[p] for p in ps)...)
end

"""
    entrepreneur_top_stats(res; p=0.01)

Compute entrepreneur representation within the top `p` fraction of the asset distribution.

Returns a NamedTuple with two versions:
- `pop`: computed over the full population
- `work`: computed over working ages only (`h < RetAge`)

For each version, it returns:
- `overall_ent_wealth_share`: entrepreneurs' share of total assets in the group
- `cutoff`: wealth cutoff defining the top `p`
- `ent_pop_share_top`: entrepreneur mass share within the top group
- `ent_wealth_share_top`: entrepreneur wealth share within the top group
"""
function entrepreneur_top_stats(res::EqmResults; p::Float64 = 0.01)
    function _calc(; working_only::Bool)
        DBN = res.DBN
        pol = res.pol
        g   = res.g
        H, na, nz, ne = size(DBN)

        totW = 0.0
        entW = 0.0
        mass_a = zeros(Float64, na)

        @inbounds for h in 1:H
            if working_only && !(h < res.p.RetAge)
                continue
            end
            for ia in 1:na, iz in 1:nz, ep in 1:ne
                m = DBN[h, ia, iz, ep]
                m == 0.0 && continue
                a = g.agrid[ia]
                totW += a * m
                mass_a[ia] += m
                if pol.e[h, ia, iz, ep] == 1
                    entW += a * m
                end
            end
        end

        overall_ent_wealth_share = (totW > 0) ? entW / totW : NaN

        W = sum(mass_a)
        cutoff = (W > 0) ? weighted_quantile(g.agrid, mass_a, 1.0 - p) : NaN

        topM = 0.0
        topW = 0.0
        topEntM = 0.0
        topEntW = 0.0

        @inbounds for h in 1:H
            if working_only && !(h < res.p.RetAge)
                continue
            end
            for ia in 1:na, iz in 1:nz, ep in 1:ne
                m = DBN[h, ia, iz, ep]
                m == 0.0 && continue
                a = g.agrid[ia]
                a >= cutoff || continue
                topM += m
                topW += a * m
                if pol.e[h, ia, iz, ep] == 1
                    topEntM += m
                    topEntW += a * m
                end
            end
        end

        ent_pop_share_top    = (topM > 0) ? topEntM / topM : NaN
        ent_wealth_share_top = (topW > 0) ? topEntW / topW : NaN

        return (
            overall_ent_wealth_share = overall_ent_wealth_share,
            cutoff = cutoff,
            ent_pop_share_top = ent_pop_share_top,
            ent_wealth_share_top = ent_wealth_share_top
        )
    end

    return (pop = _calc(working_only=false), work = _calc(working_only=true))
end

"""
    bequest_stats(res; qs=[0.50, 0.90, 0.99])

Compute bequest-flow moments using the stationary DBN and policy `ia_next`.

Definitions:
- Gross bequest for a dying parent is `b_gross = a_next`.
- Net bequest uses `Inheritance.net_bequest(b_gross, bip_util)` with `transfer=0`
  to keep warm-glow conventions consistent.

Returns:
- `Bgross_flow`, `Bnet_flow`, `TaxRev_flow`
- `Bgross_over_K`, `Bnet_over_K`
- `deaths_mass`
- `percentiles_gross` as a Dict mapping each q to its gross bequest percentile among deaths
"""
function bequest_stats(res::EqmResults; qs = [0.50, 0.90, 0.99])
    p   = res.p
    g   = res.g
    pol = res.pol
    DBN = res.DBN
    bip = res.bip

    H, na, nz, ne = size(DBN)

    bip_util = Inheritance.BequestInheritanceParams(τ_b=bip.τ_b, exemption=bip.exemption, transfer=0.0)

    Bgross = 0.0
    Bnet   = 0.0
    TaxRev = 0.0
    deaths_mass = 0.0

    w_beq = zeros(Float64, na)

    @inbounds for h in 1:H
        s = (h < H) ? p.survP[h] : 0.0
        death_prob = 1.0 - s
        death_prob <= 0.0 && continue

        for ia in 1:na, iz in 1:nz, ep in 1:ne
            m = DBN[h, ia, iz, ep]
            m == 0.0 && continue

            ia′ = pol.ia_next[h, ia, iz, ep]
            b_gross = g.agrid[ia′]

            dm = death_prob * m
            deaths_mass += dm

            Bgross += b_gross * dm
            w_beq[ia′] += dm

            taxable = max(0.0, b_gross - bip.exemption)
            tax = bip.τ_b * taxable
            TaxRev += tax * dm

            b_net = Inheritance.net_bequest(b_gross, bip_util)
            Bnet += b_net * dm
        end
    end

    K = res.agg.K_supply
    Bgross_over_K = _safe_div(Bgross, K)
    Bnet_over_K   = _safe_div(Bnet, K)

    pct = Dict{Float64,Float64}()
    for q in qs
        pct[q] = weighted_quantile(g.agrid, w_beq, q)
    end

    return (
        deaths_mass = deaths_mass,
        Bgross_flow = Bgross,
        Bnet_flow   = Bnet,
        TaxRev_flow = TaxRev,
        Bgross_over_K = Bgross_over_K,
        Bnet_over_K   = Bnet_over_K,
        percentiles_gross = pct
    )
end

"""
    cev_ratio(V0, V1, σ)

Compute the consumption-equivalent variation (CEV) from the value ratio:

`CE = (V1/V0)^(1/(1-σ)) - 1`

Returns `NaN` if `V0` or `V1` is non-finite, `V0==0`, or the two values have opposite signs.
"""
function cev_ratio(V0::Float64, V1::Float64, σ::Float64)
    (!isfinite(V0) || !isfinite(V1) || V0 == 0.0) && return NaN
    sign(V0) != sign(V1) && return NaN
    return (V1 / V0)^(1.0 / (1.0 - σ)) - 1.0
end

"""
    welfare_CE_by_age(res0, res1; weights_source=:baseline)

Compute CEV by age group.

If `weights_source=:baseline`, use `res0.DBN` as weights for both worlds.
If `weights_source=:reform`, use `res1.DBN` as weights.
"""
function welfare_CE_by_age(res0::EqmResults, res1::EqmResults;
                           weights_source::Symbol = :baseline)
    σ = res0.p.σ
    DBNw = (weights_source == :reform) ? res1.DBN : res0.DBN

    H, na, nz, ne = size(DBNw)

    CE    = fill(NaN, H)
    V0bar = fill(NaN, H)
    V1bar = fill(NaN, H)

    @inbounds for h in 1:H
        den  = 0.0
        num0 = 0.0
        num1 = 0.0

        for ia in 1:na, iz in 1:nz, ep in 1:ne
            w = DBNw[h, ia, iz, ep]
            w == 0.0 && continue
            den  += w
            num0 += res0.V[h, ia, iz, ep] * w
            num1 += res1.V[h, ia, iz, ep] * w
        end

        v0 = den > 0 ? num0 / den : NaN
        v1 = den > 0 ? num1 / den : NaN

        V0bar[h] = v0
        V1bar[h] = v1
        CE[h]    = cev_ratio(v0, v1, σ)
    end

    return (CE = CE, V0bar = V0bar, V1bar = V1bar)
end

"""
    welfare_CE_by_skillbins(res0, res1; bins, weights_source=:baseline)

Compute CEV by skill/ability bins defined as vectors of `iz` indices.

If `weights_source=:baseline`, use `res0.DBN` as weights for both worlds.
If `weights_source=:reform`, use `res1.DBN` as weights.
"""
function welfare_CE_by_skillbins(res0::EqmResults, res1::EqmResults;
                                 bins::Vector{Vector{Int}},
                                 weights_source::Symbol = :baseline)
    σ = res0.p.σ
    DBNw = (weights_source == :reform) ? res1.DBN : res0.DBN

    H, na, nz, ne = size(DBNw)
    nb = length(bins)

    CE   = fill(NaN, nb)
    V0b  = fill(NaN, nb)
    V1b  = fill(NaN, nb)

    @inbounds for b in 1:nb
        den  = 0.0
        num0 = 0.0
        num1 = 0.0

        for iz in bins[b], h in 1:H, ia in 1:na, ep in 1:ne
            w = DBNw[h, ia, iz, ep]
            w == 0.0 && continue
            den  += w
            num0 += res0.V[h, ia, iz, ep] * w
            num1 += res1.V[h, ia, iz, ep] * w
        end

        v0 = den > 0 ? num0 / den : NaN
        v1 = den > 0 ? num1 / den : NaN

        V0b[b] = v0
        V1b[b] = v1
        CE[b]  = cev_ratio(v0, v1, σ)
    end

    return (CE = CE, V0bar = V0b, V1bar = V1b)
end

"""
    welfare_CE_by_wealthbins(res0, res1; qs=[0,0.5,0.9,0.99,1], weights_source=:baseline)

Compute CEV by wealth bins defined by asset quantiles in the weighting distribution.

The cutoffs are computed from the pooled cross-sectional asset distribution implied by `DBNw`.
"""
function welfare_CE_by_wealthbins(res0::EqmResults, res1::EqmResults;
                                  qs::Vector{Float64} = [0.0, 0.5, 0.9, 0.99, 1.0],
                                  weights_source::Symbol = :baseline)

    σ = res0.p.σ
    DBNw = (weights_source == :reform) ? res1.DBN : res0.DBN
    g = res0.g

    H, na, nz, ne = size(DBNw)

    avals = Float64[]
    awts  = Float64[]
    sizehint!(avals, H * na * nz ÷ 2)
    sizehint!(awts,  H * na * nz ÷ 2)

    @inbounds for h in 1:H, ia in 1:na, iz in 1:nz, ep in 1:ne
        m = DBNw[h, ia, iz, ep]
        m == 0.0 && continue
        push!(avals, g.agrid[ia])
        push!(awts,  m)
    end

    cuts = [weighted_quantile(avals, awts, q) for q in qs]
    nb = length(qs) - 1

    CE  = fill(NaN, nb)
    V0b = fill(NaN, nb)
    V1b = fill(NaN, nb)

    @inbounds for b in 1:nb
        lo = cuts[b]
        hi = cuts[b+1]

        den  = 0.0
        num0 = 0.0
        num1 = 0.0

        for h in 1:H, ia in 1:na, iz in 1:nz, ep in 1:ne
            m = DBNw[h, ia, iz, ep]
            m == 0.0 && continue
            a = g.agrid[ia]
            in_bin = (a >= lo) && (b == nb ? (a <= hi) : (a < hi))
            in_bin || continue
            den  += m
            num0 += res0.V[h, ia, iz, ep] * m
            num1 += res1.V[h, ia, iz, ep] * m
        end

        v0 = den > 0 ? num0 / den : NaN
        v1 = den > 0 ? num1 / den : NaN

        V0b[b] = v0
        V1b[b] = v1
        CE[b]  = cev_ratio(v0, v1, σ)
    end

    return (CE = CE, V0bar = V0b, V1bar = V1b, qs = qs, cuts = cuts)
end

"""
    welfare_CE_by_occupation(res0, res1; occ_source=:baseline, weights_source=:baseline)

Compute CEV for workers (`e=0`) and entrepreneurs (`e=1`).

`occ_source` determines which policy (`res0.pol` or `res1.pol`) defines the occupation groups.
`weights_source` determines which DBN (`res0.DBN` or `res1.DBN`) provides the weights.
"""
function welfare_CE_by_occupation(res0::EqmResults, res1::EqmResults;
                                  occ_source::Symbol = :baseline,
                                  weights_source::Symbol = :baseline)

    σ = res0.p.σ
    DBNw = (weights_source == :reform) ? res1.DBN : res0.DBN
    polO = (occ_source == :reform) ? res1.pol : res0.pol

    H, na, nz, ne = size(DBNw)

    function group_ce(e_val::Int)
        den  = 0.0
        num0 = 0.0
        num1 = 0.0

        @inbounds for h in 1:H, ia in 1:na, iz in 1:nz, ep in 1:ne
            w = DBNw[h, ia, iz, ep]
            w == 0.0 && continue
            polO.e[h, ia, iz, ep] == e_val || continue
            den  += w
            num0 += res0.V[h, ia, iz, ep] * w
            num1 += res1.V[h, ia, iz, ep] * w
        end

        v0 = den > 0 ? num0 / den : NaN
        v1 = den > 0 ? num1 / den : NaN
        return (CE = cev_ratio(v0, v1, σ), V0bar = v0, V1bar = v1)
    end

    return (worker = group_ce(0),
            entrepreneur = group_ce(1),
            occ_source = occ_source,
            weights_source = weights_source)
end

end