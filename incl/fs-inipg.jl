module IniPG

    using DataFrames
    using Random
    using StatsModels
    using ..DataMod, ..Utils

    export inipg

    mutable struct Particle
        position::Vector{Float64}
        velocity::Vector{Float64}
        pbest::Vector{Float64}
        pbest_fit::Float64
        pbest_nvars::Integer
    end

    mutable struct Swarm
        particles::Vector{Particle}
        gbest::Vector{Float64}
        gbest_fit::Float64
        gbest_nvars::Integer
        gbest_index::Integer
        Swarm() = new(Vector{Particle}(), Vector{Float64}(), Inf, typemax(Int), -1)
    end
    Base.getindex(S::Swarm, i::Int)::Particle = getindex(S.particles, i)
    Base.iterate(S::Swarm) = iterate(S.particles)
    Base.iterate(S::Swarm, state) = iterate(S.particles, state)
    Base.push!(S::Swarm, p::Particle) = push!(S.particles, p)
    Base.length(S::Swarm) = length(S.particles)

    count_selected_vars(position::Vector{Float64}, θ::Float64)::Integer =
        sum(position .> θ)
    count_selected_vars(part::Particle, θ::Float64)::Integer =
        count_selected_vars(part.position, θ)

    get_selected_vars(position::Vector{Float64}, xs::Vector{Symbol}, θ::Float64)::Vector{Symbol} =
        xs[position .> θ]
    get_selected_vars(part::Particle, xs::Vector{Symbol}, θ::Float64)::Vector{Symbol} =
        get_selected_vars(part.position, xs, θ)

    function fitness(position::Vector{Float64}, data::Data, θ::Float64)::Float64
        selected = get_selected_vars(position, data.xs, θ)
        models = get_models(data, selected)
        return prediction_error(models, data)
    end
    fitness(part::Particle, data::Data, θ::Float64)::Float64 =
        fitness(part.position, data, θ)
    
    function set_gbest!(swarm::Swarm, verbose::Bool)::Swarm
        minind = argmin([p.pbest_fit for p in swarm])
        bestpart = @inbounds swarm[minind]
        if bestpart.pbest_fit < swarm.gbest_fit || (bestpart.pbest_fit == swarm.gbest_fit && bestpart.pbest_nvars < swarm.gbest_nvars)
            verbose && println("gbest updated -- gbest is now particle #$minind")
            swarm.gbest = bestpart.pbest
            swarm.gbest_fit = bestpart.pbest_fit
            swarm.gbest_nvars = bestpart.pbest_nvars
            swarm.gbest_index = minind
        end
        return swarm
    end

    function initialize_swarm(n::Integer, data::Data, rng::AbstractRNG, θ::Float64, low_frac::Float64,
                              low_var_frac::Float64, high_var_frac_range::Tuple{Float64, Float64}, verbose::Bool)::Swarm
        swarm = Swarm()
        nlow = n * low_frac
        nvars = length(data.xs)
        nvarlow = round(Int, low_var_frac * nvars)
        @inbounds for i in 1:n
            nsel = i <= nlow ? nvarlow : rand(rng, round(Int, high_var_frac_range[1] * nvars):round(Int, high_var_frac_range[2] * nvars))
            vals = shuffle(rng, [(1 - θ) * rand(rng, nsel) .+ θ; θ * rand(rng, nvars - nsel)])
            pbest_fit = fitness(vals, data, θ)
            pbest_nvars = count_selected_vars(vals, θ)
            part = Particle(vals, zeros(nvars), vals, pbest_fit, pbest_nvars)
            push!(swarm, part)
        end
        set_gbest!(swarm, verbose)
        return swarm
    end

    function pso_loop!(swarm::Swarm, data::Data, rng::AbstractRNG, θ::Float64,
                       max_iter::Integer, inertia::Float64, c1::Float64, c2::Float64, verbose::Bool)::Swarm
        for i in 1:max_iter
            verbose && println("PSO iteration $i")
            r1 = rand(rng)
            r2 = rand(rng)
            p = 1
            for part in swarm
                part.velocity = inertia * part.velocity + c1 * r1 * (part.pbest - part.position) + c2 * r2 * (swarm.gbest - part.position)
                part.position += part.velocity
                score = fitness(part, data, θ)
                nvars = count_selected_vars(part, θ)
                if score < part.pbest_fit || (score == part.pbest_fit && nvars < part.pbest_nvars)
                    verbose && println("pbest updated #$p")
                    part.pbest = part.position
                    part.pbest_fit = score
                    part.pbest_nvars = nvars
                end
                p += 1
            end
            set_gbest!(swarm, verbose)
        end
        return swarm
    end

    function inipg(data::Data, rng::Union{AbstractRNG, Nothing} = nothing;
                   n::Integer = 30, θ::Float64 = 0.6, low_frac::Float64 = 2/3, low_var_frac::Float64 = 0.1,
                   high_var_frac_range::Tuple{Float64, Float64} = (0.5, 1.0), max_iter::Integer = 100,
                   inertia::Float64 = 0.7298, c1::Float64 = 1.49618, c2::Float64 = 1.49618, verbose::Bool = true)::Vector{RegressionModel}
        isnothing(rng) && (rng = MersenneTwister())
        swarm = initialize_swarm(n, data, rng, θ, low_frac, low_var_frac, high_var_frac_range, verbose)
        pso_loop!(swarm, data, rng, θ, max_iter, inertia, c1, c2, verbose)
        verbose && println("Best solution was found by particle #$(swarm.gbest_index)")
        return get_models(data, get_selected_vars(swarm.gbest, data.xs, θ))
    end

end

using .IniPG
export inipg
