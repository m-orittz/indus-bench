module Distribution

export stationary_distribution

using Base.Threads

import ..Parameters
import ..Grids
import ..Household
import ..Inheritance
import ..Progress

"""
    _max_abs_diff(A, B)

Return `maximum(abs.(A .- B))` for 4D arrays without allocating.
Used as a convergence metric for the stationary distribution iteration.
"""
@inline function _max_abs_diff(A::Array{Float64,4}, B::Array{Float64,4})
    H, na, nz, ne = size(A)
    m = 0.0
    @inbounds for ep in 1:ne, iz in 1:nz, ia in 1:na, h in 1:H
        d = abs(A[h, ia, iz, ep] - B[h, ia, iz, ep])
        d > m && (m = d)
    end
    return m
end

"""
    stationary_distribution(p, g, pol;
        bip=BequestInheritanceParams(),
        init_DBN=nothing,
        max_iter=800,
        tol=1e-8,
        α=0.5)

Compute the stationary distribution over `(age h, asset index ia, ability index iz,
previous occupation ep)`.

Model features:
- Ability `z` is fixed within life (no within-life transition).
- Dynastic inheritance of `z` across generations via a Markov matrix `Pz`.
- Newborn assets equal net-of-tax bequests from deceased parents.
- `ep ∈ {1,2}`: 1 = was worker last period, 2 = was entrepreneur last period.

Mass conservation:
- Survivors move to `(h+1, ia′, iz, e_chosen+1)` where `e_chosen = pol.e[h,ia,iz,ep]`.
- Deaths become newborns at `(h=1, ia_b, iz_child, ep=1)` — newborns have no prior occupation.

Notes:
- Requires `pol.ia_next[h,ia,iz,ep]` and `pol.e[h,ia,iz,ep]`.
- Optional damping `α` improves convergence for dynastic problems.
"""
function stationary_distribution(p::Parameters.ModelParameters,
                                g::Grids.ModelGrids,
                                pol::Household.Policies;
                                bip::Inheritance.BequestInheritanceParams = Inheritance.BequestInheritanceParams(),
                                init_DBN = nothing,
                                max_iter::Int = 800,
                                tol::Float64  = 1e-8,
                                α::Float64    = 0.5)

    H  = p.MaxAge
    na = length(g.agrid)
    nz = length(g.zgrid)

    # === Dynastic z transition matrix (across generations only) ===
    shp = Inheritance.SkillInheritanceParams(ρ=p.ρz, σ_ε=p.σz_eps, μ=p.μ_z)
    Pz  = Inheritance.skill_transition_matrix(g.zgrid, shp)

    # === Initial guess ===
    DBN = zeros(Float64, H, na, nz, 2)
    if init_DBN !== nothing
        @assert size(init_DBN) == (H, na, nz, 2) "init_DBN has wrong shape."
        DBN .= init_DBN
        DBN ./= sum(DBN)
    else
        # Newborns start with ep=1 (no prior occupation). Seed at (h=1, ia=1, iz, ep=1).
        Gz0 = Inheritance.stationary_dist(Pz)
        @inbounds for iz in 1:nz
            DBN[1, 1, iz, 1] = Gz0[iz]
        end
        DBN ./= sum(DBN)
    end

    # === Precompute newborn asset index implied by a parent's chosen next-asset index ia′ ===
    beq_ia = Vector{Int}(undef, na)
    @inbounds for ia′ in 1:na
        b_net = Inheritance.net_bequest(g.agrid[ia′], bip)
        beq_ia[ia′] = Grids.nearest_index(g.agrid, b_net)
    end

    # === Preallocation for speed + thread-safety ===
    nt = Threads.maxthreadid()
    DBN_locals = [zeros(Float64, H, na, nz, 2) for _ in 1:nt]
    DBN_new    = zeros(Float64, H, na, nz, 2)

    pb_dbn = Progress.ProgressBar("Stationary distribution", max_iter; show_eta=true)

    for iter in 1:max_iter
        Progress.update!(pb_dbn, iter)
        # === reset locals ===
        for t in 1:nt
            fill!(DBN_locals[t], 0.0)
        end

        # === Thread over parent ia: safe via thread-local arrays ===
        # Parallelize over asset grid (na=100) instead of ability grid (nz=9) for better thread utilization
        Threads.@threads for ia in 1:na
            tid = Threads.threadid()
            DBNt = DBN_locals[tid]

            @inbounds for h in 1:H
                s_hp1 = (h < H) ? p.survP[h] : 0.0

                for iz in 1:nz
                    for ep in 1:2
                        mass = DBN[h, ia, iz, ep]
                        mass == 0.0 && continue

                        ia′      = pol.ia_next[h, ia, iz, ep]
                        e_chosen = pol.e[h, ia, iz, ep]   # 0 or 1

                        # === survivors age up: chosen occupation becomes new ep ===
                        if h < H && s_hp1 > 0.0
                            DBNt[h+1, ia′, iz, e_chosen + 1] += s_hp1 * mass
                        end

                        # === deaths become newborns: always ep=1 (no prior occupation) ===
                        death_mass = (1.0 - s_hp1) * mass
                        if death_mass > 0.0
                            ia_b = beq_ia[ia′]
                            for izc in 1:nz
                                DBNt[1, ia_b, izc, 1] += death_mass * Pz[iz, izc]
                            end
                        end
                    end
                end
            end
        end

        # === combine locals into DBN_new ===
        fill!(DBN_new, 0.0)
        @inbounds for t in 1:nt
            L = DBN_locals[t]
            for ep in 1:2, iz in 1:nz, ia in 1:na, h in 1:H
                DBN_new[h, ia, iz, ep] += L[h, ia, iz, ep]
            end
        end

        # === normalize (numerical drift) ===
        s = sum(DBN_new)
        @assert s > 0.0 "DBN_new mass is zero; check transitions."
        DBN_new ./= s

        # === damping ===
        if α < 1.0
            @inbounds @. DBN_new = α * DBN_new + (1 - α) * DBN
            DBN_new ./= sum(DBN_new)
        end

        diff = _max_abs_diff(DBN_new, DBN)
        if diff < tol
            Progress.finish!(pb_dbn, "converged")
            return DBN_new
        end

        # === swap without allocating ===
        DBN, DBN_new = DBN_new, DBN
    end

    Progress.finish!(pb_dbn, "max iterations")
    return DBN
end

end
