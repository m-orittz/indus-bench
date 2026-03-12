module Aggregates

export compute_aggregates

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

    @inbounds for h in 1:H
        working = (h < p.RetAge)

        for ia in 1:na
            a = g.agrid[ia]
            for iz in 1:nz
                for ep in 1:ne
                    mass = DBN[h, ia, iz, ep]
                    mass == 0.0 && continue

                    Pop_mass += mass
                    K_supply += a * mass

                    if working
                        WorkPop_mass += mass

                        if pol.e[h, ia, iz, ep] == 0
                            # === worker ===
                            L += mass
                            pension_tax_base_w += w * mass

                        else
                            # === entrepreneur (working-age only) ===
                            Ent_mass += mass

                            k = pol.k[h, ia, iz, ep]
                            K_demand += k * mass
                            Debt     += max(k - a, 0.0) * mass

                            if k > 0.0
                                z = g.zgrid[iz]

                                # === Q aggregation term ===
                                Q_sum += Production.entrepreneur_q_term(z, k, p) * mass

                                # === profit consistent with Household ===
                                π = Production.entrepreneur_profit(z, k, p, prices)
                                pension_tax_base_pi += max(π, 0.0) * mass

                                # === MPK diagnostic ===
                                MPK_i = Production.entrepreneur_mpk(z, k, p)
                                r_i   = MPK_i - δ
                                rK_sum      += r_i * mass
                                Ent_cap_mass += mass
                            end
                        end
                    else
                        # === retired ===
                        RetPop_mass += mass
                    end
                end
            end
        end
    end

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