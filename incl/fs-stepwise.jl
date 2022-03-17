module Stepwise

    using DataFrames
    using StatsBase
    using ..DataMod, ..Utils

    export stepwise

    # The code for `step` and `stepwise` below is inspired by the code by Bogumił Kamiński, found at
    # https://stackoverflow.com/questions/49794476/backward-elimination-forward-selection-in-multilinear-regression-in-julia

    function multibic(data::Data, rhs::Vector{Symbol})::Tuple{Float64, Vector{RegressionModel}}
        models = get_models(data, rhs)::Vector{RegressionModel}
        return (mean(bic.(models))::Float64, models)
    end
    function multibic(models::Vector{RegressionModel})::Float64
        return mean(bic.(models))
    end

    function step(data::Data, rhs::Vector{Symbol}, forward::Bool,
                  prevbest::Float64, prevmodel::Vector{RegressionModel},
                  verbose::Bool, debug::Bool)::Tuple{Vector{Symbol}, Bool, Float64, Vector{RegressionModel}}
        options = forward ? setdiff(data.xs, rhs) : rhs
        if isempty(options)
            return (rhs, false, prevbest, prevmodel)
        end
        bestbic = prevbest
        bestrhs = rhs
        bestmodel = prevmodel
        improved = false
        for opt in options
            thisrhs = forward ? [rhs; opt] : setdiff(rhs, opt)
            thisbic, thismodel = multibic(data, thisrhs)
            if debug
                print(opt); print(" - "); println(thisbic)
            end
            if thisbic < bestbic
                bestbic = thisbic
                bestrhs = thisrhs
                bestmodel = thismodel
                improved = true
            end
        end
        if verbose
            if improved
                println("$(forward ? "forward" : "backward") step - new rhs: $bestrhs\nwith mean BIC $bestbic")
            else
                println("$(forward ? "forward" : "backward") step could not improve BIC")
            end
        end
        return (bestrhs, improved, bestbic, bestmodel)
    end

    function stepwise(data::Data; verbose::Bool = true, debug::Bool = false)::Vector{RegressionModel}
        rhs = Symbol[]
        model = get_models(data, Symbol[])
        bestbic = multibic(model)
        verbose && println("BIC empty model: $bestbic")
        while true
            verbose && println("step")
            rhs, improvedfwd, bestbic, model = step(data, rhs, true, bestbic, model, verbose, debug)
            rhs, improvedbwd, bestbic, model = step(data, rhs, false, bestbic, model, verbose, debug)
            if !improvedfwd && !improvedbwd
                sort!(rhs)
                verbose && println("final rhs: $rhs")
                return model
            end
        end
    end

end

using .Stepwise
export stepwise
