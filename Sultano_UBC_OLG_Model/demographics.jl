module Demographics

"""
    bell_miller_pop()

Returns the Bell–Miller (2002) population-by-age vector used in Guvenen et al.
Hard-coded for MaxAge = 81, matching the Fortran code.
"""
function bell_miller_pop()
    MaxAge = 81
    pop = zeros(Float64, MaxAge)

    pop[1]  = 197_316.0
    pop[2]  = 197_141.0
    pop[3]  = 196_959.0
    pop[4]  = 196_770.0
    pop[5]  = 196_580.0
    pop[6]  = 196_392.0
    pop[7]  = 196_205.0
    pop[8]  = 196_019.0
    pop[9]  = 195_830.0
    pop[10] = 195_634.0
    pop[11] = 195_429.0
    pop[12] = 195_211.0
    pop[13] = 194_982.0
    pop[14] = 194_739.0
    pop[15] = 194_482.0
    pop[16] = 194_211.0
    pop[17] = 193_924.0
    pop[18] = 193_619.0
    pop[19] = 193_294.0
    pop[20] = 192_945.0
    pop[21] = 192_571.0
    pop[22] = 192_169.0
    pop[23] = 191_736.0
    pop[24] = 191_271.0
    pop[25] = 190_774.0
    pop[26] = 190_243.0
    pop[27] = 189_673.0
    pop[28] = 189_060.0
    pop[29] = 188_402.0
    pop[30] = 187_699.0
    pop[31] = 186_944.0
    pop[32] = 186_133.0
    pop[33] = 185_258.0
    pop[34] = 184_313.0
    pop[35] = 183_290.0
    pop[36] = 182_181.0
    pop[37] = 180_976.0
    pop[38] = 179_665.0
    pop[39] = 178_238.0
    pop[40] = 176_689.0
    pop[41] = 175_009.0
    pop[42] = 173_187.0
    pop[43] = 171_214.0
    pop[44] = 169_064.0
    pop[45] = 166_714.0
    pop[46] = 164_147.0
    pop[47] = 161_343.0
    pop[48] = 158_304.0
    pop[49] = 155_048.0
    pop[50] = 151_604.0
    pop[51] = 147_990.0
    pop[52] = 144_189.0
    pop[53] = 140_180.0
    pop[54] = 135_960.0
    pop[55] = 131_532.0
    pop[56] = 126_888.0
    pop[57] = 122_012.0
    pop[58] = 116_888.0
    pop[59] = 111_506.0
    pop[60] = 105_861.0
    pop[61] = 99_957.0
    pop[62] = 93_806.0
    pop[63] = 87_434.0
    pop[64] = 80_882.0
    pop[65] = 74_204.0
    pop[66] = 67_462.0
    pop[67] = 60_721.0
    pop[68] = 54_053.0
    pop[69] = 47_533.0
    pop[70] = 41_241.0
    pop[71] = 35_259.0
    pop[72] = 29_663.0
    pop[73] = 24_522.0
    pop[74] = 19_890.0
    pop[75] = 15_805.0
    pop[76] = 12_284.0
    pop[77] = 9_331.0
    pop[78] = 6_924.0
    pop[79] = 5_016.0
    pop[80] = 3_550.0
    pop[81] = 2_454.0 

    return pop
end

"""
    compute_survP(pop)

Given a population-by-age vector `pop`, returns the conditional survival
probabilities survP[h] = P(alive at h+1 | alive at h), with survP[end] = 0.0.
"""
function compute_survP(pop::AbstractVector{<:Real})
    MaxAge = length(pop)
    survP = Vector{Float64}(undef, MaxAge)
    for age in 1:MaxAge-1
        survP[age] = pop[age+1] / pop[age]
    end
    survP[MaxAge] = 0.0
    return survP
end

"""
    bell_miller_pop_surv()

Convenience wrapper: returns `(pop, survP)` for Bell–Miller demographics.
"""
function bell_miller_pop_surv()
    pop = bell_miller_pop()
    survP = compute_survP(pop)
    return pop, survP
end

end