module Household

using Base.Threads

import ..Parameters
import ..Grids
import ..Prices
import ..Utility
import ..Production
import ..Inheritance

export Policies, solve_household

"""
    Policies

Policy functions stored on the state grid (h, ia, iz, ep):

- `c[h, ia, iz, ep]`: consumption
- `ia_next[h, ia, iz, ep]`: next-period asset index on the grid
- `e[h, ia, iz, ep]`: occupation choice (0=worker, 1=entrepreneur)
- `k[h, ia, iz, ep]`: entrepreneurial capital choice (0 for workers)

The fourth dimension `ep ∈ {1, 2}` encodes the previous-period occupation:
- `ep = 1`: was a worker last period (pays entry cost F if switching to entrepreneur)
- `ep = 2`: was an entrepreneur last period (no entry cost)
"""
struct Policies
    c::Array{Float64,4}
    ia_next::Array{Int,4}
    e::Array{Int,4}
    k::Array{Float64,4}
end

"""
    is_retired(p, h)

Return `true` if age `h` is in retirement (forced retirement starts at `p.RetAge`).
"""
@inline is_retired(p::Parameters.ModelParameters, h::Int) = (h >= p.RetAge)

"""
    pension_tax(p)

Convenience helper returning the pension contribution rate τ_pension.
"""
@inline pension_tax(p::Parameters.ModelParameters) = p.τ_pension

"""
    tax_positive_income(y, τ)

Apply a proportional tax `τ` to positive income only:
`y_net = y - τ*max(y,0)`.
"""
@inline tax_positive_income(y::Float64, τ::Float64) = y - τ * max(y, 0.0)

"""
    worker_income(p, prices, h)

Worker income (net of pension tax):

- Working ages: `(1-τ_pension) * w`
- Retired ages: flat pension benefit `b_pension`
"""
@inline function worker_income(p::Parameters.ModelParameters, prices::Prices.ModelPrices, h::Int)
    if is_retired(p, h)
        return p.b_pension
    else
        τ = pension_tax(p)
        return (1.0 - τ) * prices.w
    end
end

"""
    solve_household(p, g, prices; bip=BequestInheritanceParams())

Solve the household problem by backward induction with occupational choice
`e ∈ {0,1}`, mortality, warm-glow bequests, and a one-time entry cost `p.F`.

State: (age h, assets a, ability z, previous occupation ep).
- ep=1: was worker last period → pays entry cost F when choosing entrepreneur.
- ep=2: was entrepreneur last period → no entry cost.

The continuation value is indexed by the *chosen* occupation (which becomes
next period's ep), so two distinct V_survive values are read per savings choice.

Notes:
- Retirement is forced: no entrepreneurship for `h >= p.RetAge`.
- Bequest utility uses `transfer=0` (to avoid putting newborn transfers inside warm-glow).
- With `p.F = 0`, results are identical to a model without entry costs.
"""
function solve_household(p::Parameters.ModelParameters,
                         g::Grids.ModelGrids,
                         prices::Prices.ModelPrices;
                         bip::Inheritance.BequestInheritanceParams = Inheritance.BequestInheritanceParams())

    # === Warm-glow uses transfer=0 convention (keeps utility "net-of-tax" only). ===
    bip_util = Inheritance.BequestInheritanceParams(τ_b=bip.τ_b, exemption=bip.exemption, transfer=0.0)

    H  = p.MaxAge
    na = length(g.agrid)
    nz = length(g.zgrid)

    V      = fill(Utility.BIG_NEG, H, na, nz, 2)
    c_pol  = zeros(H, na, nz, 2)
    ia_pol = ones(Int, H, na, nz, 2)
    e_pol  = zeros(Int, H, na, nz, 2)
    k_pol  = zeros(H, na, nz, 2)

    r  = prices.r
    τp = pension_tax(p)

    # === Precompute warm-glow value for each asset grid point interpreted as gross bequest. ===
    vbq = Vector{Float64}(undef, na)
    @inbounds for ia′ in 1:na
        b = Inheritance.net_bequest(g.agrid[ia′], bip_util)
        vbq[ia′] = Utility.v_bequest(b, p)
    end

    # =========================================================
    # === Terminal age H ===
    # =========================================================
    h = H
    retiredH = is_retired(p, h)
    income_H = worker_income(p, prices, h)

    for ep in 1:2
        entry_cost = (ep == 1) ? p.F : 0.0

        Threads.@threads for iz in 1:nz
            @inbounds begin
                z = g.zgrid[iz]

                for ia in 1:na
                    a = g.agrid[ia]

                    res_W = (1.0 + r) * a + income_H

                    res_E   = -Inf
                    k_opt   = 0.0
                    if !retiredH
                        π_E, k_opt = Production.entrepreneur_profit_k(a, z, iz, p, prices)
                        π_E_net    = tax_positive_income(π_E, τp)
                        res_E      = (1.0 + r) * a + π_E_net - entry_cost
                    end

                    res_max = retiredH ? res_W : max(res_W, res_E)
                    ia_max  = clamp(searchsortedlast(g.agrid, res_max), 1, na)

                    best_val = Utility.BIG_NEG
                    best_c   = 0.0
                    best_ia′ = 1
                    best_e   = 0
                    best_k   = 0.0

                    for ia′ in 1:ia_max
                        a_next = g.agrid[ia′]
                        vb     = vbq[ia′]

                        cW   = res_W - a_next
                        valW = Utility.u(cW, p) + p.β * vb

                        if retiredH
                            val, e_choice, c_choice, k_choice = valW, 0, cW, 0.0
                        else
                            cE   = res_E - a_next
                            valE = Utility.u(cE, p) + p.β * vb

                            if valE > valW
                                val, e_choice, c_choice, k_choice = valE, 1, cE, k_opt
                            else
                                val, e_choice, c_choice, k_choice = valW, 0, cW, 0.0
                            end
                        end

                        if val > best_val
                            best_val, best_c, best_ia′, best_e, best_k = val, c_choice, ia′, e_choice, k_choice
                        end
                    end

                    V[h, ia, iz, ep]      = best_val
                    c_pol[h, ia, iz, ep]  = best_c
                    ia_pol[h, ia, iz, ep] = best_ia′
                    e_pol[h, ia, iz, ep]  = best_e
                    k_pol[h, ia, iz, ep]  = best_k
                end
            end
        end
    end  # ep loop (terminal age)

    # =========================================================
    # === Backwards for h = H-1,...,1 ===
    # =========================================================
    for h in (H-1):-1:1
        s_hp1   = p.survP[h]
        retired = is_retired(p, h)
        income_h = worker_income(p, prices, h)

        for ep in 1:2
            entry_cost = (ep == 1) ? p.F : 0.0

            Threads.@threads for iz in 1:nz
                @inbounds begin
                    z = g.zgrid[iz]

                    for ia in 1:na
                        a = g.agrid[ia]

                        res_W = (1.0 + r) * a + income_h

                        # === Compute entrepreneur option once (only if not retired) ===
                        res_E   = -Inf
                        k_opt   = 0.0
                        if !retired
                            π_E, k_opt = Production.entrepreneur_profit_k(a, z, iz, p, prices)
                            π_E_net    = tax_positive_income(π_E, τp)
                            res_E      = (1.0 + r) * a + π_E_net - entry_cost
                        end

                        res_max = retired ? res_W : max(res_W, res_E)
                        ia_max  = clamp(searchsortedlast(g.agrid, res_max), 1, na)

                        best_val = Utility.BIG_NEG
                        best_c   = 0.0
                        best_ia′ = 1
                        best_e   = 0
                        best_k   = 0.0

                        for ia′ in 1:ia_max
                            a_next = g.agrid[ia′]

                            # Continuation value depends on the occupation *chosen* this period,
                            # which becomes next period's ep (ep=1 if worker, ep=2 if entrepreneur).
                            V_survive_W = V[h+1, ia′, iz, 1]   # chose worker → ep_next = 1
                            V_survive_E = V[h+1, ia′, iz, 2]   # chose entrepreneur → ep_next = 2
                            V_die       = vbq[ia′]

                            cW   = res_W - a_next
                            valW = Utility.u(cW, p) + p.β * (s_hp1 * V_survive_W + (1 - s_hp1) * V_die)

                            if retired
                                val, e_choice, c_choice, k_choice = valW, 0, cW, 0.0
                            else
                                cE   = res_E - a_next
                                valE = Utility.u(cE, p) + p.β * (s_hp1 * V_survive_E + (1 - s_hp1) * V_die)

                                if valE > valW
                                    val, e_choice, c_choice, k_choice = valE, 1, cE, k_opt
                                else
                                    val, e_choice, c_choice, k_choice = valW, 0, cW, 0.0
                                end
                            end

                            if val > best_val
                                best_val, best_c, best_ia′, best_e, best_k = val, c_choice, ia′, e_choice, k_choice
                            end
                        end

                        V[h, ia, iz, ep]      = best_val
                        c_pol[h, ia, iz, ep]  = best_c
                        ia_pol[h, ia, iz, ep] = best_ia′
                        e_pol[h, ia, iz, ep]  = best_e
                        k_pol[h, ia, iz, ep]  = best_k
                    end
                end
            end
        end  # ep loop
    end  # h loop

    return V, Policies(c_pol, ia_pol, e_pol, k_pol)
end

end
