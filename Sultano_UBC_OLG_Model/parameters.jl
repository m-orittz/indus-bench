module Parameters

export ModelParameters, baseline_parameters, with

import ..Demographics

"""
    ModelParameters

Container for structural parameters of the OLG–entrepreneurship model.
"""
Base.@kwdef struct ModelParameters
    # === Demographics ===
    MaxAge::Int    
    RetAge::Int    
    pop::Vector{Float64}
    survP::Vector{Float64}

    # === Pension system (PAYG, fast) ===
    τ_pension::Float64 = 0.124   
    b_pension::Float64 = 0.0     

    # === Preferences ===
    β::Float64     
    σ::Float64     
    χ_bq::Float64  
    bq_0::Float64  

    # === Technology ===
    α::Float64     
    δ::Float64     
    μ::Float64     

    # === Entrepreneurial ability process (z) ===
    nz::Int
    nz_aux::Int
    ρz::Float64
    σz_eps::Float64
    μ_z::Float64
    z_scale::Float64

    # === Entrepreneurship / borrowing ===
    F::Float64
    A_e::Float64

    # === Financial constraints ===
    θ_vec::Vector{Float64}

    # === Asset grid ===
    na::Int
    fine_na::Int
    a_theta::Float64
    amax::Float64
    amin::Float64

    # === Misc numerics ===
    mtauchen::Float64
    mtauchen_z::Float64
    brent_tol::Float64
end

"""
    baseline_parameters()

Create a baseline parameter set using values from from Guvenen et al.
Some entries (F, z_scale) are placeholders that I will calibrate or fill later.
"""
function baseline_parameters()
    MaxAge = 81
    RetAge = 45

    # === Bell–Miller demographics from Demographics module ===
    pop, survP = Demographics.bell_miller_pop_surv()

    @assert length(pop) == MaxAge
    @assert length(survP) == MaxAge

    return ModelParameters(
        MaxAge = MaxAge,
        RetAge = RetAge,
        pop    = pop,
        survP  = survP,

        # === Pension (PAYG): fixed tax rate, endogenous b_pension updated in GE ===
        τ_pension = 0.124,
        b_pension = 0.0,

        # === Preferences ===
        β     = 0.9593, 
        σ     = 4.0,
        χ_bq  = 0.2,
        bq_0  = 26800,

        # === Technology ===
        α     = 0.40,
        δ     = 0.05,
        μ     = 0.90,

        # === Entrepreneurial ability process ===
        nz        = 9,
        nz_aux    = 11,
        ρz        = 0.1,
        σz_eps    = 0.277,
        μ_z       = 0.0,
        z_scale   = 1.6,  # calibration starting point: 1.1 gave 0.004% entrepreneurs (near-zero Jacobian); 2.0 should give ~5-15%

        # === Entrepreneurship ===
        F     = 0.05,
        A_e   = 0.20978768460588718,

        # === Borrowing constraints θ(z) ===
        θ_vec = [1.000,
                 1.225,
                 1.450,
                 1.675,
                 1.900,
                 2.125,
                 2.350,
                 2.575,
                 2.800],

        # === Asset grid ===
        na        = 100, #51→100 with w-FP fix (bisection→FP speedup ~20-30×)
        fine_na   = 400, #800
        a_theta   = 4.0,
        amax      = 500_000.0,
        amin      = 0.0001,

        # === Misc numerics ===
        mtauchen  = 3.0,
        mtauchen_z = 5.0,
        brent_tol = 1e-8,
    )
end

"""
    _nt(p::ModelParameters)

Internal helper: convert a `ModelParameters` instance into a `NamedTuple`
with the same fieldnames. This is used to implement `with(...)` updates
without repeating the full constructor.
"""
@inline function _nt(p::ModelParameters)
    names = fieldnames(ModelParameters)
    vals  = Tuple(getfield(p, nm) for nm in names)
    return NamedTuple{names}(vals)
end

"""
    with(p::ModelParameters; kwargs...)

Return a copy of `p` with any fields provided as keyword arguments replaced.
This is a convenience wrapper around the `ModelParameters` constructor that
avoids repeating all fields when updating a small subset.
"""
@inline function with(p::ModelParameters; kwargs...)
    base = _nt(p)
    return ModelParameters(; base..., kwargs...)
end

"""
    with_Ae(p::ModelParameters, A_e::Real)

Return a copy of p with entrepreneurial TFP A_e replaced.
"""
with_Ae(p::ModelParameters, A_e::Real) = with(p; A_e=Float64(A_e))

"""
    with_pension(p::ModelParameters; τ_pension=p.τ_pension, b_pension=p.b_pension)

Return a copy of p with pension parameters replaced.
"""
with_pension(p::ModelParameters; τ_pension::Real=p.τ_pension, b_pension::Real=p.b_pension) =
    with(p; τ_pension=Float64(τ_pension), b_pension=Float64(b_pension))


end