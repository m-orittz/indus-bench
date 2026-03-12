module Inheritance

export BequestInheritanceParams, net_bequest

import ..Parameters
import ..Grids

# =============================================================
# Normal CDF helper (kept self-contained, no external packages)
# =============================================================

"""
    Φ(x)

Standard Normal CDF approximation (Abramowitz–Stegun).
Kept internal to avoid external package dependencies.
"""
@inline function Φ(x::Real)
    if x == 0
        return 0.5
    end
    ax = abs(Float64(x))
    t  = 1.0 / (1.0 + 0.2316419 * ax)
    d  = 0.3989422804014327 * exp(-0.5 * ax * ax)  # 1/sqrt(2π)

    poly = t * (0.319381530 +
           t * (-0.356563782 +
           t * (1.781477937 +
           t * (-1.821255978 +
           t * 1.330274429))))

    p = 1.0 - d * poly
    return x >= 0 ? p : 1.0 - p
end


# =============================================================
# Skill (entrepreneurial ability) inheritance: dynastic AR(1)
# =============================================================

"""
    SkillInheritanceParams

Parameters for the dynastic skill/ability inheritance process.

We interpret the process as an AR(1) in logs across generations:

    log(z_child) = (1-ρ) * μ + ρ * log(z_parent) + ε,
    ε ~ N(0, σ_ε^2).

- ρ controls intergenerational persistence.
- σ_ε is the innovation s.d. (NOT the stationary s.d.).
- μ is the long-run mean of log z.
"""
Base.@kwdef struct SkillInheritanceParams
    ρ::Float64
    σ_ε::Float64
    μ::Float64 = 0.0
end

"""
    skill_log_cutoffs(zgrid)

Construct log-space cutoffs (length nz+1) using midpoints between adjacent grid points.
These cutoffs define the discrete states for mapping a continuous log(z) draw into
one of the nz grid points.
"""
function skill_log_cutoffs(zgrid::AbstractVector{<:Real})
    nz = length(zgrid)
    @assert nz >= 2 "zgrid must have at least 2 points"

    logz = log.(Float64.(zgrid))
    cuts = Vector{Float64}(undef, nz + 1)
    cuts[1] = -Inf
    for j in 2:nz
        cuts[j] = 0.5 * (logz[j-1] + logz[j])
    end
    cuts[end] = Inf
    return cuts
end

"""
    skill_transition_matrix(zgrid, shp)

Build an nz×nz Markov transition matrix Pz where

    Pz[i,j] = Pr(z_child = zgrid[j] | z_parent = zgrid[i]).

The mapping uses midpoint cutoffs in log-space.
"""
function skill_transition_matrix(zgrid::AbstractVector{<:Real},
                                 shp::SkillInheritanceParams)
    @assert shp.σ_ε > 0.0 "σ_ε must be > 0 for a meaningful transition matrix."

    nz = length(zgrid)
    logz = log.(Float64.(zgrid))
    cuts = skill_log_cutoffs(zgrid)

    Pz = zeros(Float64, nz, nz)
    for i in 1:nz
        m = (1 - shp.ρ) * shp.μ + shp.ρ * logz[i]
        for j in 1:nz
            a = (cuts[j]   - m) / shp.σ_ε
            b = (cuts[j+1] - m) / shp.σ_ε
            Pz[i, j] = Φ(b) - Φ(a)
        end
        # numerical safety
        rowsum = sum(Pz[i, :])
        rowsum != 0.0 && (Pz[i, :] ./= rowsum)
    end

    return Pz
end

"""
    stationary_dist(P; tol=1e-14, max_iter=100_000)

Compute the stationary distribution π of a Markov matrix P using power iteration:

    π_{t+1} = π_t * P.

Returns π as a length-n row-vector (ordinary Vector).
"""
function stationary_dist(P::AbstractMatrix{<:Real};
                         tol::Float64 = 1e-14,
                         max_iter::Int = 100_000)
    n = size(P, 1)
    @assert size(P, 2) == n "P must be square"

    π = fill(1.0 / n, n)
    for _ in 1:max_iter
        π_new = vec(π' * P)

        s = sum(π_new)
        s != 0.0 && (π_new ./= s)

        if maximum(abs.(π_new .- π)) < tol
            return π_new
        end
        π = π_new
    end
    return π
end

"""
    stationary_skill_dist(zgrid, p; method=:ergodic)

Convenience wrapper returning the stationary cross-sectional mass Gz over `zgrid`
implied by the dynastic inheritance process in `p`.

- method=:ergodic (default): discretize the dynastic AR(1) and compute π(Pz).
- method=:normal: use the continuous stationary distribution of log(z) implied by the AR(1)
  and integrate it over the log-midpoint cutoffs.
"""
function stationary_skill_dist(zgrid::AbstractVector{<:Real},
                               p::Parameters.ModelParameters;
                               method::Symbol = :ergodic)
    shp = SkillInheritanceParams(ρ=p.ρz, σ_ε=p.σz_eps, μ=p.μ_z)

    if method == :ergodic
        Pz = skill_transition_matrix(zgrid, shp)
        Gz = stationary_dist(Pz)

    elseif method == :normal
        @assert abs(shp.ρ) < 1.0 "Require |ρ|<1 for a stationary AR(1) distribution."
        @assert shp.σ_ε > 0.0   "σ_ε must be > 0."

        cuts = skill_log_cutoffs(zgrid)
        σ_stat = shp.σ_ε / sqrt(1 - shp.ρ^2)

        nz = length(zgrid)
        Gz = Vector{Float64}(undef, nz)
        for j in 1:nz
            a = (cuts[j]   - shp.μ) / σ_stat
            b = (cuts[j+1] - shp.μ) / σ_stat
            Gz[j] = Φ(b) - Φ(a)
        end
        Gz ./= sum(Gz)

    else
        error("Unknown method = $method. Use method=:ergodic or :normal")
    end

    @assert length(Gz) == length(zgrid)
    @assert isapprox(sum(Gz), 1.0; atol=1e-12)
    return Gz
end


# =============================================================
# Bequest inheritance: net-of-tax mapping
# =============================================================

"""
    BequestInheritanceParams

Parameters for a bequest/inheritance transfer rule.

- `τ_b`: flat inheritance tax rate on taxable bequests.
- `exemption`: amount exempt from taxation (tax applies only above it).
- `transfer`: lump-sum transfer added to each newborn (e.g., UBC financed elsewhere).
"""
Base.@kwdef struct BequestInheritanceParams
    τ_b::Float64 = 0.0
    exemption::Float64 = 0.0
    transfer::Float64 = 0.0
end

"""
    net_bequest(b_gross, bip)

Compute net inheritance received by the child from a gross bequest `b_gross`.
"""
function net_bequest(b_gross::Real, bip::BequestInheritanceParams)
    b = max(0.0, Float64(b_gross))
    taxable = max(0.0, b - bip.exemption)
    tax = bip.τ_b * taxable
    return max(0.0, b - tax + bip.transfer)
end

"""
    bequest_to_asset_index(agrid, b_net)

Map a newborn's net inheritance `b_net` into an asset grid index.
Uses `Grids.nearest_index` (same convention used elsewhere).
"""
bequest_to_asset_index(agrid::AbstractVector{<:Real}, b_net::Real) =
    Grids.nearest_index(agrid, b_net)

end