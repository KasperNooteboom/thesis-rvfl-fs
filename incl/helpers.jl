module Helpers

using CSV
using DataFrames
using Formatting
using StatsBase
using StatsModels

export get_coefs, allequal, featurestrings
export ys1d, roundclip, f1, log2file, xdecim

get_coefs(models::Vector{<:RegressionModel})::Vector{Vector{Float64}} = coef.(models)
function allequal(x::Vector)
    length(x) < 2 && return true
    e1 = x[1]
    @inbounds for i in 2:length(x)
        x[i] == e1 || return false
    end
    return true
end
function featurestrings(models::Vector{<:RegressionModel})::Vector{String}
    coefs = get_coefs(models)
    selected = [c .!= 0 for c in coefs]
    if allequal(selected)
        return [join([b ? "1" : "0" for b in selected[1]])]
    else
        return [join([b ? "1" : "0" for b in s]) for s in selected]
    end
end

ys1d(ymat::VecOrMat{Int})::Vector{Int} = ymat
function ys1d(ymat::Matrix{Float64})::Vector{Int}
    if size(ymat, 2) > 1
        [argmax(row) for row in eachrow(ymat)]
    else
        vec(1 * (ymat .>= .5))
    end
end
roundclip(vec::Vector{Float64}, range::UnitRange{Int})::Vector{Int} = max.(first(range), min.(last(range), round.(Int, vec)))

xdecim(num::Real, dec::Int) = sprintf1("%.$(dec)f", num)

function log2file(filename::String, x::String)
    path = "log/$filename.txt"
    dir, _ = splitdir(path)
    mkpath(dir)
    open(path, "a") do io
        write(io, "$x\n")
    end
end

function log2file(filename::String, df::DataFrame, header::Bool = false)
    CSV.write("log/$filename.txt", df, append = true, writeheader = header)
end
log2file(filename::String, mat::Matrix, names::Vector, header::Bool = false) = log2file(filename, DataFrame(mat, names), header)

end

using .Helpers
