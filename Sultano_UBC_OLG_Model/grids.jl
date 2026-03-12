module Grids

export ModelGrids, build_grids, nearest_index

import ..Parameters

"""
    ModelGrids

Container for the discretized state-space grids used in the model.

Fields
- `agrid`: asset grid (Float64 vector)
- `zgrid`: entrepreneurial ability grid (Float64 vector)
- `Gz`: stationary cross-sectional probability mass over `zgrid`
"""
struct ModelGrids
    agrid::Vector{Float64}
    zgrid::Vector{Float64}
    Gz::Vector{Float64}
end

"""
    build_grids(p::Parameters.ModelParameters)

Construct and return the model grids given parameters `p`.

- Builds a convex asset grid on [`p.amin`, `p.amax`] using curvature `p.a_theta`.
- Loads the 9-point entrepreneurial ability grid (from Guvenen et al.) and rescales it by `p.z_scale`.
- Loads the stationary cross-sectional mass `Gz` over the ability grid.
"""
function build_grids(p::Parameters.ModelParameters)
    # === Asset grid ===
    agrid_raw = range(p.amin^(1 / p.a_theta), p.amax^(1 / p.a_theta), length=p.na)
    agrid = agrid_raw .^ p.a_theta

    # === Entrepreneurial ability grid (Guvenen et al.), rescaled by z_scale ===
    zgrid = [
        0.42979057335122889,
        0.57304504679830393,
        0.75699738889794321,
        1.0000000000000000,
        1.3210085195350887,
        1.7450635086842872,
        2.3052437621017381,
        3.0452466493415149,
        4.0227967678658256,
    ]
    zgrid .= p.z_scale .* zgrid

    @assert length(zgrid) == p.nz "length(zgrid) != p.nz"

    # === Stationary cross-sectional mass over Guvenen 9-point zgrid ===
    Gz = [
        0.005923815772741314,
        0.060883385496116715,
        0.2417303374571288,
        0.3829249225480262,
        0.2417303374571289,
        0.060597535943081926,
        0.005977036246740619,
        0.00022923140591080138,
        0.0000033976731247387093,
    ]
    @assert length(Gz) == length(zgrid)
    @assert isapprox(sum(Gz), 1.0; atol=1e-12)

    return ModelGrids(agrid, zgrid, Gz)
end

"""
    nearest_index(xgrid, x)

Return the index of the grid point in `xgrid` closest to `x`.
"""
function nearest_index(xgrid::AbstractVector{<:Real}, x::Real)
    g = Float64.(xgrid)
    xx = Float64(x)
    idx = searchsortedfirst(g, xx)
    if idx <= 1
        return 1
    elseif idx > length(g)
        return length(g)
    else
        return abs(g[idx] - xx) < abs(g[idx-1] - xx) ? idx : idx-1
    end
end

end 