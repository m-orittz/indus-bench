module Production

export FinalGoodResults,
       final_output,
       mpL,
       mpQ,
       final_good_block,
       implied_Ae,
       k_star,
       entrepreneur_profit,
       entrepreneur_mpk,
       entrepreneur_q_term,
       entrepreneur_profit_k

import ..Parameters
import ..Prices
import ..Utility

"""
    FinalGoodResults

Results from the final-good (Cobb–Douglas) block.

Fields
- `Y`: final output
- `w`: wage (marginal product of labor)
- `pQ`: price of intermediate aggregate `Q` (marginal product of `Q`)
"""
struct FinalGoodResults
    Y::Float64
    w::Float64
    pQ::Float64
end

const _FLOOR = 1e-14

"""
    final_output(Q, L, α)

Final-good production function:
`Y = Q^α * L^(1-α)`.
"""
@inline final_output(Q::Real, L::Real, α::Real) = Q^α * L^(1 - α)

"""
    mpL(Q, L, α)

Marginal product of labor:
`w = ∂Y/∂L`.
"""
@inline mpL(Q::Real, L::Real, α::Real) = (1 - α) * Q^α * L^(-α)

"""
    mpQ(Q, L, α)

Marginal product of `Q`:
`pQ = ∂Y/∂Q`.
"""
@inline mpQ(Q::Real, L::Real, α::Real) = α * Q^(α - 1) * L^(1 - α)

"""
    implied_Ae(pQ, Q, μ)

Competitive part in the monopolistic-competition block:
`A_e = pQ * Q^(1-μ)`.

Uses a small floor for `Q` to avoid domain errors.
"""
@inline implied_Ae(pQ::Real, Q::Real, μ::Real) = pQ * (max(Float64(Q), _FLOOR))^(1 - μ)

"""
    final_good_block(Q, L, α)

Compute `(Y, w, pQ)` given `(Q, L, α)` and return a `FinalGoodResults`.

Uses small floors for `Q` and `L` to avoid domain errors.
"""
function final_good_block(Q::Real, L::Real, α::Real)
    Qeff = max(Float64(Q), _FLOOR)
    Leff = max(Float64(L), _FLOOR)

    Y  = final_output(Qeff, Leff, α)
    w  = mpL(Qeff, Leff, α)
    pQ = mpQ(Qeff, Leff, α)

    return FinalGoodResults(Y, w, pQ)
end

"""
    final_good_block(p, agg)

Convenience wrapper using parameter `p.α` and aggregates `agg.Q`, `agg.L`.
"""
@inline final_good_block(p, agg) = final_good_block(agg.Q, agg.L, p.α)


# ============================================================
# Entrepreneur technology block (canonical)
# ============================================================

"""
    k_star(z, p, prices)

Unconstrained optimal capital demand for an entrepreneur with ability `z`
given prices (interest rate `r`) and parameters.
"""
@inline function k_star(z::Float64, p::Parameters.ModelParameters, prices::Prices.ModelPrices)
    μ   = p.μ
    den = prices.r + p.δ
    num = μ * p.A_e * (z^μ)

    if z <= 0.0 || !(num > 0.0) || !(den > 0.0)
        return 0.0
    end
    return (num / den)^(1.0 / (1.0 - μ))
end

"""
    entrepreneur_q_term(z, k, p)

Intermediate-good “quantity” term used in aggregation:
`(z*k)^μ`.
"""
@inline entrepreneur_q_term(z::Float64, k::Float64, p::Parameters.ModelParameters) = (z * k)^(p.μ)

"""
    entrepreneur_profit(z, k, p, prices)

Per-period entrepreneurial profit:
`π = A_e (z k)^μ - (r+δ)k`.

Note: `p.F` is a one-time entry cost paid when first becoming an entrepreneur.
      It is handled in the household budget constraint (household.jl), not here.
"""
@inline function entrepreneur_profit(z::Float64, k::Float64,
                                    p::Parameters.ModelParameters,
                                    prices::Prices.ModelPrices)
    rev  = p.A_e * (z * k)^p.μ
    cost = (prices.r + p.δ) * k
    return rev - cost
end

"""
    entrepreneur_mpk(z, k, p)

Marginal product of capital in the entrepreneur technology:
`∂(A_e (z k)^μ)/∂k = μ A_e z^μ k^(μ-1)`.
"""
@inline function entrepreneur_mpk(z::Float64, k::Float64, p::Parameters.ModelParameters)
    return p.μ * p.A_e * (z^p.μ) * (k^(p.μ - 1.0))
end

"""
    entrepreneur_profit_k(a, z, iz, p, prices)

Return entrepreneurial profit `π(a,z)` and chosen capital `k(a,z)` under the
borrowing constraint:

`k ≤ θ(iz) * a`.

Returns `(Utility.BIG_NEG, 0.0)` if infeasible.
"""
@inline function entrepreneur_profit_k(a::Float64, z::Float64, iz::Int,
                                      p::Parameters.ModelParameters,
                                      prices::Prices.ModelPrices)
    θ     = p.θ_vec[iz]
    k_max = θ * a
    if k_max <= 0.0
        return Utility.BIG_NEG, 0.0
    end

    k_uncon = k_star(z, p, prices)
    k_opt   = min(k_uncon, k_max)
    if k_opt <= 0.0
        return Utility.BIG_NEG, 0.0
    end

    π = entrepreneur_profit(z, k_opt, p, prices)
    return π, k_opt
end

end