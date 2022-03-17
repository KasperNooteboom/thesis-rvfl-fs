module Lasso

    using DataFrames
    using Lasso
    using LinearAlgebra
    using StatsModels
    using ..DataMod, ..ManualModelMod, ..Utils

    export lasso, sparse_encoded_lasso
    export MinAIC, MinAICc, MinBIC, MinCVmse, MinCV1se, AllSeg

    function getonemodel(y::Symbol, xs::Vector{Symbol}, df::DataFrame, λs::Vector{Float64}, segselect::Type{<:SegSelect})::RegressionModel
        fit(LassoModel, compose(y, xs, intercept = false), df, λ = λs, standardize = false, cd_maxiter = typemax(Int), select = segselect())
    end
    function getfullpath(y::Symbol, xs::Vector{Symbol}, df::DataFrame, λs::Vector{Float64})
        fit(LassoPath, compose(y, xs, intercept = false), df, λ = λs, standardize = false, cd_maxiter = typemax(Int))
    end

    function lasso(data::Data, λs::Vector{Float64} = .5 .^ collect(-5:14);
                   fullpath::Bool = false, xs::Vector{Symbol} = data.xs,
                   segselect::Type{<:SegSelect} = MinBIC)
        if fullpath
            return RegressionModel[getfullpath(y, xs, data.df, λs) for y in data.ys]
        else
            return RegressionModel[getonemodel(y, xs, data.df, λs, segselect) for y in data.ys]
        end
    end

    function sparse_encoded_lasso(data::Data, λs::Vector{Float64} = .5 .^ collect(-5:14))::Vector{ManualModel}
        models = lasso(data, λs)
        deselected = [coef(model) .== 0 for model in models]
        len = eachindex(data.ys)
        nobs = nrow(data.df)
        dat = [hcat(ones(Float64, nobs), data.xmat) for _ in len]
        empt = zeros(nobs)
        @inbounds for i in len
            dat[i][:, deselected[i]] .= empt
        end
        mp = [pinv(mat) for mat in dat]
        yvec = [data.df[!, y] for y in data.ys]
        wts = mp .* yvec
        return [ManualModel(wts[i], true, data.ys[i], data) for i in len]::Vector{ManualModel}
    end

end

using .Lasso
export lasso, sparse_encoded_lasso
export MinAIC, MinAICc, MinBIC, MinCVmse, MinCV1se, AllSeg
