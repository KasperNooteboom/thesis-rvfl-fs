module FS

using ..DataMod, ..Helpers
using MultivariateStats, Statistics, StatsBase, StatsModels

include("./manualmodel.jl")
export ManualModel, predict

module Utils
    using DataFrames, LinearAlgebra, StatsBase, StatsModels, ..DataMod, ..Helpers, ..ManualModelMod
    function compose(y::Symbol, terms::Vector{Symbol}; intercept::Bool = true)::FormulaTerm
        return term(y) ~ foldl(+, term.(intercept ? [1; terms] : terms))
    end
    function lt_featurename(f1::Symbol, f2::Symbol)
        s1, s2 = string(f1), string(f2)
        if s1[1] == s2[1]
            return isless(parse(Int, s1[2:end]), parse(Int, s2[2:end]))
        else
            return isless(s2[1], s1[1])
        end
    end
    function get_model(data::Data, sel_xmat::Matrix{Float64}, y_vec::Vector{Float64},
                       desel_i::Vector{Int}, predicted::Symbol, intercept::Bool)::RegressionModel
        # Simply doing `sel_xmat \ y_vec` resulted in a rare error when
        # sel_xmat is square. The \ operator interprets this incorrectly.
        # Therefore, this line was extracted from the \ operator's code:
        weights = qr(sel_xmat, ColumnNorm()) \ y_vec
        weights = copy(weights)
        for i in desel_i
            insert!(weights, intercept ? i + 1 : i, 0.0)
        end
        return ManualModel(weights, intercept, predicted, data)
    end
    function get_models(data::Data, rhs::Vector{Symbol}, intercept::Bool = true)::Vector{RegressionModel}
        desel_i = findall(x -> x ∉ rhs, data.xs)
        n = nrow(data.df)
        sort!(rhs, lt = lt_featurename)
        sel_xmat = isempty(rhs) ? zeros(Float64, n, 0) : (data.xs == rhs ? data.xmat : Matrix(select(data.xdf, rhs)))::Matrix{Float64}
        intercept && (sel_xmat = hcat(ones(Float64, n), sel_xmat))
        y_vecs = Vector{Float64}[data.ydf[!, y] for y in data.ys]
        return map(i -> get_model(data, sel_xmat, y_vecs[i], desel_i, data.ys[i], intercept), 1:length(y_vecs))
    end
    Base.iterate(s::Symbol) = (s, nothing)
    Base.iterate(::Symbol, ::Any) = nothing
    Base.length(s::Symbol) = 1
    export compose, get_models
end
using .Utils
export compose, get_models

include("./fs-stepwise.jl")
include("./fs-lasso.jl")
include("./fs-inipg.jl")
include("./fs-importance.jl")
include("./fs-ga.jl")
include("./fs-l1l2reg.jl")

llsqreg(data::Data)::Vector{RegressionModel} = get_models(data, data.xs)
function ridgereg(data::Data, λs::Vector{Float64} = .5 .^ collect(-5:14))::Vector{RegressionModel}
    ridges = [ridge(data.xmat, data.ymat, λ) for λ in λs]
    c = 1:length(data.ys)
    splits = [[ begin
                    y = r[:, i]
                    [y[end]; y[1:end-1]]
                end for i in c] for r in ridges]
    models = [[ManualModel(split[i], true, data.ys[i], data) for i in c] for split in splits]
    bics = [mean([bic(model[i]) for i in c]) for model in models]
    minind = argmin(bics)
    return models[minind]
end
export llsqreg, ridgereg

end

using .FS
