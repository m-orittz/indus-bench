module Utility

export BIG_NEG, u, v_bequest

import ..Parameters

"""
    BIG_NEG

Large negative number used as a proxy for “minus infinity” when utility
should rule out infeasible choices (e.g., nonpositive consumption).
"""
const BIG_NEG = -1.0e16

"""
    u(c::Float64, p::Parameters.ModelParameters)

CRRA period utility from consumption.

Returns `BIG_NEG` if `c <= 0` to rule out infeasible consumption.
"""
@inline function u(c::Float64, p::Parameters.ModelParameters)
    if c <= 0.0
        return BIG_NEG
    end
    σ = p.σ
    s = 1 - σ
    return c^Int(s) / s
end

"""
    v_bequest(b::Float64, p::Parameters.ModelParameters)

Warm-glow bequest utility:
`v(b) = χ_bq * (b + bq_0)^(1-σ) / (1-σ)`.

- Allows `b >= 0` (including `b = 0`).
- Returns `BIG_NEG` if `b < 0`.
- If `χ_bq == 0`, returns `0.0`.
"""
@inline function v_bequest(b::Float64, p::Parameters.ModelParameters)
    χ  = p.χ_bq
    σ  = p.σ
    b0 = p.bq_0

    χ == 0.0 && return 0.0
    b < 0.0 && return BIG_NEG

    btilde = b + b0

    if σ == 1.0
        return χ * log(btilde)
    else
        s = 1 - σ
        return χ * btilde^Int(s) / s
    end
end

end 
