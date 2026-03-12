module Aggregates

export compute_aggregates

using Base.Threads

import ..Prices
import ..Parameters
import ..Grids
import ..Household
import ..Production

"""
    compute_aggregates(p, g, pol, DBN, prices)

Given the stationary distribution `DBN[h, ia, iz]`, policy functions `pol`,
grids `g`, and prices `prices`, compute model aggregates.

Key distinction (as in Guvenen et al.):
- `K_supply`: aggregate household assets/wealth ∫ a_i di (supply of funds)
- `K_demand`: unadjusted capital stock ∫ k_i di used in intermediate production

Working-age objects use `h < p.RetAge`:
- `L`: mass of workers (working-age only)
- `Ent_mass`: mass of entrepreneurs (working-age only)
- `Ent_share_work`: entrepreneurs / working-age population
- `Ent_share_pop`: entrepreneurs / total population

Intermediate-good composite:
- `Q = (∑ (z*k)^μ * mass)^(1/μ)` over working-age entrepreneurs with `k>0`
- `TFP_Q = Q / K_demand`

Pension system (PAYG, fixed τ, endogenous b):
- tax base = wages of workers + positive entrepreneurial profits (working ages only)
- revenue  = τ_pension * tax base
- implied benefit per retiree: `b_pension_implied = revenue / RetPop`
"""
function compute_aggregates(p::Parameters.ModelParameters,
                           g::Grids.ModelGrids,
                           pol::Household.Policies,
                           DBN::Array{Float64,4},
                           prices::Prices.ModelPrices)

    H, na, nz, ne = size(DBN)

    K_supply = 0.0
    K_demand = 0.0
    Debt     = 0.0

    L            = 0.0
    Ent_mass     = 0.0
    Pop_mass     = 0.0
    WorkPop_mass = 0.0
    RetPop_mass  = 0.0

    μ = p.μ
    δ = p.δ
    w = prices.w

    Q_sum       = 0.0
    rK_sum      = 0.0
    Ent_cap_mass = 0.0

    # === Pension tax base components (working ages only) ===
    pension_tax_base_w  = 0.0
    pension_tax_base_pi = 0.0

    # === Thread-local accumulators for parallel reduction ===
    nt = Threads.nthreads()
    local_K_supply = zeros(Float64, nt)
    local_K_demand = zeros(Float64, nt)
    local_Debt = zeros(Float64, nt)
    local_L = zeros(Float64, nt)
    local_Ent_mass = zeros(Float64, nt)
    local_Pop_mass = zeros(Float64, nt)
    local_WorkPop_mass = zeros(Float64, nt)
    local_RetPop_mass = zeros(Float64, nt)
    local_Q_sum = zeros(Float64, nt)
    local_rK_sum = zeros(Float64, nt)
    local_Ent_cap_mass = zeros(Float64, nt)
    local_pension_tax_base_w = zeros(Float64, nt)
    local_pension_tax_base_pi = zeros(Float64, nt)

    # Parallelize over age h (81 ages) for better thread utilization
    @inbounds Threads.@threads for h in 1:H
        tid = Threads.threadid()
        working = (h < p.RetAge)

        for ia in 1:na
            a = g.agrid[ia]
            for iz in 1:nz
                for ep in 1:ne
                    mass = DBN[h, ia, iz, ep]
                    mass == 0.0 && continue

                    local_Pop_mass[tid] += mass
                    local_K_supply[tid] += a * mass

                    if working
                        local_WorkPop_mass[tid] += mass

                        if pol.e[h, ia, iz, ep] == 0
                            # === worker ===
                            local_L[tid] += mass
                            local_pension_tax_base_w[tid] += w * mass

                        else
                            # === entrepreneur (working-age only) ===
                            local_Ent_mass[tid] += mass

                            k = pol.k[h, ia, iz, ep]
                            local_K_demand[tid] += k * mass
                            local_Debt[tid]     += max(k - a, 0.0) * mass

                            if k > 0.0
                                z = g.zgrid[iz]

                                # === Q aggregation term ===
                                local_Q_sum[tid] += Production.entrepreneur_q_term(z, k, p) * mass

                                # === profit consistent with Household ===
                                π = Production.entrepreneur_profit(z, k, p, prices)
                                local_pension_tax_base_pi[tid] += max(π, 0.0) * mass

                                # === MPK diagnostic ===
                                MPK_i = Production.entrepreneur_mpk(z, k, p)
                                r_i   = MPK_i - δ
                                local_rK_sum[tid]      += r_i * mass
                                local_Ent_cap_mass[tid] += mass
                            end
                        end
                    else
                        # === retired ===
                        local_RetPop_mass[tid] += mass
                    end
                end
            end
        end
    end

    # === Reduce thread-local accumulators ===
    K_supply = sum(local_K_supply)
    K_demand = sum(local_K_demand)
    Debt = sum(local_Debt)
    L = sum(local_L)
    Ent_mass = sum(local_Ent_mass)
    Pop_mass = sum(local_Pop_mass)
    WorkPop_mass = sum(local_WorkPop_mass)
    RetPop_mass = sum(local_RetPop_mass)
    Q_sum = sum(local_Q_sum)
    rK_sum = sum(local_rK_sum)
    Ent_cap_mass = sum(local_Ent_cap_mass)
    pension_tax_base_w = sum(local_pension_tax_base_w)
    pension_tax_base_pi = sum(local_pension_tax_base_pi)

    Pop_mass <= 0.0 && error("Population mass is non-positive; check DBN.")

    K_supply_per_cap = K_supply / Pop_mass
    K_demand_per_cap = K_demand / Pop_mass

    L_share_pop   = L / Pop_mass
    Ent_share_pop = Ent_mass / Pop_mass

    Ent_share_work = (WorkPop_mass > 0.0) ? (Ent_mass / WorkPop_mass) : 0.0
    L_share_work   = (WorkPop_mass > 0.0) ? (L / WorkPop_mass) : 0.0

    Q           = (Q_sum > 0.0) ? Q_sum^(1.0 / μ) : 0.0
    TFP_Q       = (K_demand > 0.0 && Q > 0.0) ? Q / K_demand : 0.0
    TFP_Q_assets = (K_supply > 0.0 && Q > 0.0) ? Q / K_supply : 0.0
    rK_mean     = (Ent_cap_mass > 0.0) ? rK_sum / Ent_cap_mass : 0.0

    # === Pension revenue and implied benefit (balanced by construction for fixed τ) ===
    τp = p.τ_pension
    pension_revenue    = τp * (pension_tax_base_w + pension_tax_base_pi)
    b_pension_implied  = (RetPop_mass > 0.0) ? pension_revenue / RetPop_mass : 0.0

    return (
        K_supply = K_supply,
        K_demand = K_demand,
        Debt     = Debt,

        L        = L,
        Pop      = Pop_mass,
        WorkPop  = WorkPop_mass,
        RetPop   = RetPop_mass,

        K_supply_per_cap = K_supply_per_cap,
        K_demand_per_cap = K_demand_per_cap,

        L_share_pop      = L_share_pop,
        Ent_share_pop    = Ent_share_pop,
        L_share_work     = L_share_work,
        Ent_share_work   = Ent_share_work,

        Q            = Q,
        TFP_Q        = TFP_Q,
        TFP_Q_assets = TFP_Q_assets,
        rK_mean      = rK_mean,

        # === Pension objects ===
        pension_tax_base_w  = pension_tax_base_w,
        pension_tax_base_pi = pension_tax_base_pi,
        pension_revenue     = pension_revenue,
        b_pension_implied   = b_pension_implied
    )
end

end