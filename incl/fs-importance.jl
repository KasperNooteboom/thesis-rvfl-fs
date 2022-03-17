module Importance
    
    using DataFrames
    using Statistics
    using StatsBase
    using StatsModels
    using ..DataMod, ..Lasso, ..Utils

    include("./correlation_drop.jl")

    export importance, importance_list, importance_corfilter

    function r2lasso(path::RegressionModel)::Vector{Float64}
        y = path.model.m.rr.y # actual Y values
        ȳ = path.mf.f.lhs.mean # mean of actual Y values
        ŷs = eachcol(predict(path)) # predicted Y values per λ
        SSreds = [mean((y - ŷ).^2) for ŷ in ŷs] # sum of squares of residuals per λ
        SStot = mean((y .- ȳ).^2) # total sum of squares (prop. to variance of the data)
        r2s = 1 .- SSreds ./ SStot # R² values per λ
        return r2s
    end

    function imp_scores_path(path::RegressionModel)::Vector{Float64}
        rsq = r2lasso(path)
        βs = eachrow(coef(path))
        scores = @inbounds Float64[sum([β[i] != 0 ? rsq[i] : 0.0 for i in eachindex(β)]) for β in βs]
        return scores
    end

    function imp_scores(paths::Vector{RegressionModel})::Vector{Tuple{Symbol, Float64}}
        ywise = imp_scores_path.(paths)
        scores = map(mean, zip(ywise...))::Vector{Float64}
        sym = @inbounds Symbol[t.sym for t in paths[1].mf.f.rhs.terms]
        return collect(zip(sym, scores))
    end

    function filter_correlations(data::Data, t::Float64)::Vector{Symbol}
        cormat = cor(data.xmat)
        drop = drop_cor(cormat, data.xs, t)
        return setdiff(data.xs, drop)
    end

    function importance_list(data::Data, cor_filter::Bool = false, t::Float64 = 0.95)::Vector{Tuple{Symbol, Float64}}
        xs = data.xs
        # Filter correlations
        if cor_filter
            xs = filter_correlations(data, t)
            data = Data(data, xs)
        end
        # Run Lasso
        paths = lasso(data, fullpath = true)
        # Get importance
        scores = imp_scores(paths)
        return sort(scores, by = z -> z[2], rev = true)
    end

    function importance(data::Data; cor_filter::Bool = false, t::Float64 = 0.95,
                        top::Float64 = 0.5)::Vector{RegressionModel}
        # Get scores
        scores = importance_list(data, cor_filter, t)
        # Select top x
        nx = length(data.xs)
        nsel = min(ceil(Int, nx * top), nx)
        selected = [s[1] for s in scores[1:nsel]]
        return get_models(data, selected)
    end

    importance_corfilter(data::Data; kwargs...)::Vector{RegressionModel} = importance(data; cor_filter = true, kwargs...)
    
end

using .Importance
export importance, importance_list, importance_corfilter
