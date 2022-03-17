module RVFLMod

using DataFrames
using LinearAlgebra
using Random
using ..DataMod

export RVFL, enhanced, radbas

randomrange(rng::AbstractRNG, C::Number, dims::Integer...)::VecOrMat{Float64} =
    2C * rand(rng, dims...) .- C
enames(n::Integer)::Vector{Symbol} = Symbol.('e', 1:n)

# RVFL Input layer
struct RVFL
    m::Integer
    A::Matrix{Float64}
    b::Vector{Float64}
    g::Function

    RVFL(in::Integer, enh::Integer, rng::AbstractRNG = MersenneTwister(); range::Float64 = 1.0, act::Function=radbas) =
        new(enh, randomrange(rng, range, enh, in), randomrange(rng, range, enh), act)
end

enhanced(m::RVFL, x::Union{Vector{Float64}, SubArray{Float64}}) =
    [x; map(m.g, [dot(m.A[i, :], x) for i in 1:m.m] .+ m.b)]

enhanced(m::RVFL, x::Matrix{Float64}) =
    mapslices(row -> enhanced(m, row), Matrix(x), dims = 2)

enhanced(m::RVFL, xdf::DataFrame) =
    DataFrame(enhanced(m, Matrix(xdf)), [propertynames(xdf); enames(m.m)])

enhanced(m::RVFL, d::Data) =
    Data(d, hcat(enhanced(m, d.xdf), d.ydf), [propertynames(d.xdf); enames(m.m)])

# Activation functions
radbas(x::Float64) = exp(-x^2)

end

using .RVFLMod
