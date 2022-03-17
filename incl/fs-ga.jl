module GA

    using DataFrames
    using Random
    using StatsModels
    using ..DataMod, ..Utils

    export ga, ga_uniform, ga_twopoint

    mutable struct Solution
        bits::BitVector
        fitness::Float64
        Solution(bits::BitVector) = new(bits, 0)
        Solution(bits::Vector{Bool}) = new(BitVector(bits))
    end
    function earlierzeros(S1::Solution, S2::Solution)::Bool
        @inbounds for i in eachindex(S1.bits, S2.bits)
            S1.bits[i] < S2.bits[i] && return true
            S1.bits[i] > S2.bits[i] && return false
        end
        return false
    end
    Base.getindex(S::Solution, i::Int)::Bool = getindex(S.bits, i)
    Base.iterate(S::Solution) = iterate(S.bits)
    Base.iterate(S::Solution, state) = iterate(S.bits, state)
    Base.eachindex(S::Solution) = eachindex(S.bits)
    Base.length(S::Solution) = length(S.bits)
    Base.:(==)(S1::Solution, S2::Solution) = S1.bits == S2.bits
    Base.isless(S1::Solution, S2::Solution) =
        S1.fitness == S2.fitness ?
            (sum(S1.bits) == sum(S2.bits) ?
                earlierzeros(S1, S2) :
                sum(S1.bits) > sum(S2.bits)) :
            S1.fitness < S2.fitness
    
    const Population = Vector{Solution}

    get_selected_vars(bits::BitVector, xs::Vector{Symbol}) = @inbounds xs[bits]
    get_selected_vars(sol::Solution, xs::Vector{Symbol}) = @inbounds xs[sol.bits]

    function fitness(sol::Union{Solution, BitVector}, data::Data, intercept::Bool)::Float64
        selected = get_selected_vars(sol, data.xs)
        models = get_models(data, selected, intercept)
        return prediction_accuracy(models, data)
    end
    function fitness!(sol::Solution, data::Data, intercept::Bool)::Float64
        sol.fitness = fitness(sol, data, intercept)
    end
    function fitness!(pop::Population, data::Data, intercept::Bool)::Vector{Float64}
        map(sol -> fitness!(sol, data, intercept), pop)
    end
    function fitness!((sol, changed)::Tuple{Solution, Bool}, data::Data, intercept::Bool)::Float64
        sol.fitness = changed ? fitness(sol, data, intercept) : sol.fitness
    end

    # Crossover functions

    function uniformX(p1::Solution, p2::Solution, p_recomb::Float64, rng::AbstractRNG)::Vector{Solution}
        return @inbounds [Solution([rand(rng) < p_recomb ? p1[i] : p2[i] for i in eachindex(p1)])]
    end

    function twopointX(p1::Solution, p2::Solution, ::Float64, rng::AbstractRNG)::Vector{Solution}
        len = length(p1)
        r1 = rand(rng, 1:(len - 1))
        r2 = rand(rng, (r1 + 1):len)
        return @inbounds [Solution([r1 <= i <= r2 ? p2[i] : p1[i] for i in eachindex(p1)])]
    end

    # Mutation functions

    function uniformM(p::Solution, p_mut::Float64, rng::AbstractRNG)::Vector{Tuple{Solution, Bool}}
        mut = rand(rng, length(p)) .<= p_mut
        new = p.bits .⊻ mut
        changed = any(mut)
        return [(changed ? Solution(new) : p, changed)]
    end

    # Higher level functions

    function crossover(pop::Population, data::Data, p_recomb::Float64, rng::AbstractRNG, Xoperator::Function, intercept::Bool)::Population
        spop = shuffle(rng, pop)
        Xpop = @inbounds reduce(vcat, [spop[i] == spop[i + 1] ? Solution[] : Xoperator(spop[i], spop[i + 1], p_recomb, rng)::Vector{Solution} for i in 1:2:length(spop)])
        fitness!(Xpop, data, intercept)
        return [spop; Xpop]
    end

    function mutation(pop::Population, data::Data, p_mut::Float64, rng::AbstractRNG, Moperator::Function, intercept::Bool)::Population
        Mpop = reduce(vcat, [Moperator(sol, p_mut, rng)::Vector{Tuple{Solution, Bool}} for sol in pop])
        map(tup -> fitness!(tup, data, intercept), Mpop)
        muts = @inbounds [tup[1] for tup in filter(t -> t[2], Mpop)]
        return [pop; muts]
    end

    function initialize_population(n::Integer, data::Data, rng::AbstractRNG, intercept::Bool)::Population
        nvars = length(data.xs)
        pop = [Solution(bitrand(rng, nvars)) for _ in 1:n]
        fitness!(pop, data, intercept)
        return pop
    end

    function ga_loop(pop::Population, data::Data, rng::AbstractRNG, n::Integer,
                     max_iter::Integer, max_best_iter::Integer, diversity_check_interval::Integer,
                     p_recomb::Float64, p_mut::Float64, Xoperator::Function, Moperator::Function,
                     verbose::Bool, intercept::Bool)::Solution
        best = zeros(Int, length(data.xs))
        ibest = 0
        if verbose
            println("Initial population:")
            display(pop)
        end
        # Loop until max iteration:
        @inbounds for i in 1:max_iter
            # Recombination
            Xpop = crossover(pop, data, p_recomb, rng, Xoperator, intercept)
            # Mutation
            Mpop = mutation(Xpop, data, p_mut, rng, Moperator, intercept)
            # Selection
            ## For now just select the top n
            sort!(Mpop, rev = true)
            pop = Mpop[1:n]
            if verbose
                println("Population after iteration $i:")
                display(pop)
            end
            # Check termination criteria
            ## Same best solution for `max_best_iter` iterations
            if pop[1].bits == best
                ibest += 1
            else
                best = pop[1].bits
                ibest = 0
            end
            if ibest >= max_best_iter
                verbose && println("Best solution remained the same in $max_best_iter iterations. Terminating.")
                return pop[1]
            end
            ## Check for loss of genetic diversity every `diversity_check_interval` iterations
            if i % diversity_check_interval == 0
                if length(unique([x.bits for x in pop])) == 1
                    verbose && println("All solutions in the population are identical. Terminating.")
                    return pop[1]
                end
            end
        end
        verbose && println("Max #iterations of $max_iter reached. Terminating after $i iterations.")
        return pop[1]
    end

    function ga(data::Data, rng::Union{AbstractRNG, Nothing} = nothing; n::Integer = 100,
                max_iter::Integer = 1000, max_best_iter::Integer = 200, diversity_check_interval::Integer = 5,
                p_recomb::Float64 = 0.5, p_mut::Float64 = 0.02, Xoperator::Symbol = :uniform,
                Moperator::Symbol = :uniform, verbose::Bool = true, intercept::Bool = true)::Vector{RegressionModel}
        isnothing(rng) && (rng = MersenneTwister())
        if Xoperator == :uniform
            X = uniformX
        elseif Xoperator == :twopoint
            X = twopointX
        else
            throw(ArgumentError("specified Xoperator is not defined"))
        end
        if Moperator == :uniform
            M = uniformM
        else
            throw(ArgumentError("specified Moperator is not defined"))
        end
        verbose && println("Using " * string(Xoperator) * " crossover and " * string(Moperator) * " mutation.")
        pop = initialize_population(n, data, rng, intercept)
        best = ga_loop(pop, data, rng, n, max_iter, max_best_iter, diversity_check_interval, p_recomb, p_mut, X, M, verbose, intercept)
        return get_models(data, data.xs[best.bits])
    end

    ga_uniform(data::Data, rng::Union{AbstractRNG, Nothing} = nothing; kwargs...)::Vector{RegressionModel} =
        ga(data, rng; Xoperator = :uniform, kwargs...)
    ga_twopoint(data::Data, rng::Union{AbstractRNG, Nothing} = nothing; kwargs...)::Vector{RegressionModel} =
        ga(data, rng; Xoperator = :twopoint, kwargs...)

end

using .GA
export ga, ga_uniform, ga_twopoint
