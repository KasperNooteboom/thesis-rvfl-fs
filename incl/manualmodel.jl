module ManualModelMod

using DataFrames
using StatsBase
using ..DataMod

export ManualModel, predict

struct ManualModel <: RegressionModel
    coefs::Vector{Float64}
    intercept::Bool
    predicted::Symbol
    data::Data
    ManualModel(coefs::Vector{Float64}, intercept::Bool, predicted::Symbol, data::Data) =
        @inbounds new([abs(coefs[i]) > eps(Float64) ? coefs[i] : 0. for i in eachindex(coefs)], intercept, predicted, data)
end

ncoef(m::ManualModel) = sum(m.coefs .!= 0.)

function Base.show(io::IO, m::ManualModel)
    println(io, "Manual linear model predicting :" * string(m.predicted) * " -- " * (m.intercept ? "" : "not ") * "using intercept")
    show(io, CoefTable([m.coefs], ["Coef."], m.intercept ? ["Intercept"; ":" .* string.(m.data.xs)] : ":" .* string.(m.data.xs)))
end

StatsBase.coef(m::ManualModel)::Vector{Float64} = m.coefs
function StatsBase.predict(m::ManualModel, A::Matrix{Float64})
    col = size(A, 2)
    len = length(m.coefs)
    len != (m.intercept ? col + 1 : col) && error("Model size of $len does not match matrix size of $col")
    m.intercept && (A = hcat(ones(Float64, size(A, 1)), A))
    return A * m.coefs
end
StatsBase.predict(m::ManualModel, A::DataFrame) = StatsBase.predict(m, Matrix(A))
StatsBase.predict(m::ManualModel) = predict(m, m.data.xmat)

function StatsBase.loglikelihood(m::ManualModel)
    # This function is derived from the loglikelihood function in GLM.jl
    pred = predict(m)
    act = m.data.ydf[!, m.predicted]::Vector{Float64}
    n = nrow(m.data.df)
    dev = 0.0
    @inbounds @simd for i in 1:n
        dev += abs2(act[i] - pred[i])::Float64
    end
    return -n/2 * (log(2π * dev/n) + 1)
end
StatsBase.dof(m::ManualModel) = ncoef(m) + 1
StatsBase.nobs(m::ManualModel) = nrow(m.data.df)

end

using .ManualModelMod
