module L1L2

    using DataFrames
    using LinearAlgebra
    using ..DataMod, ..ManualModelMod

    export l1l2

    # Soft thresholding
    function S(x::Float64, λ::Float64)::Float64
        if x >= λ
            return x - λ
        elseif x <= -λ
            return x + λ
        else
            return 0
        end
    end
    S(xs::Vector{Float64}, λ::Float64)::Vector{Float64} = [S(x, λ) for x in xs]
    S(xs::Matrix{Float64}, λ::Float64)::Vector{Float64} = S(vec(xs), λ)

    # Calculate next weight vector
    function nextβ(β::Vector{Float64}, τplus2ϵλ::Float64, λγ::Float64,
                   τIminusXᵀX::Matrix{Float64}, XᵀY::Vector{Float64})::Vector{Float64}
        return S(τIminusXᵀX * β + XᵀY, λγ) / τplus2ϵλ
    end

    # Calculate max iteration
    function lmax(β₁::Vector{Float64}, β₀::Vector{Float64}, κ::Float64,
                  κ₀::Float64, λ::Float64, ϵ::Float64, ξ::Float64)::Int
        # numnum = norm(β₁ - β₀) * (κ + κ₀ + 4 * ϵ * λ)
        # numdenom = (2 * κ₀ + 4 * ϵ * λ) * ξ
        # denomnum = κ + κ₀ + 4 * ϵ * λ
        # denomdenom = κ - κ₀ # THIS GOES WRONG IF κ == κ₀
        # num = log(numnum / numdenom)
        # denom = log(denomnum / denomdenom)
        # return round(Int, num / denom + 1, RoundDown)
        return 10000
    end

    function weight_loop(data::Data, λ::Float64, γ::Float64, ϵ::Float64, ξ::Float64)
        X = hcat(ones(Float64, size(data.xmat, 1)), data.xmat)
        Ys = Vector{Float64}[data.df[!, y] for y in data.ys]
        XᵀX = X'X
        τ = norm(XᵀX)
        τplus2ϵλ = τ + 2 * ϵ * λ
        λγ = λ * γ
        τIminusXᵀX = τ * I - XᵀX
        XᵀYs = [X'Y for Y in Ys]
        β₀ = zeros(length(data.xs) + 1)
        βs = [nextβ(β₀, τplus2ϵλ, λγ, τIminusXᵀX, XᵀY) for XᵀY in XᵀYs]
        κ, κ₀ = τ, τ # should change
        max_iter = maximum([lmax(β, β₀, κ, κ₀, λ, ϵ, ξ) for β in βs])::Int
        @inbounds for i in 1:max_iter
            βs = Vector{Float64}[nextβ(βs[y], τplus2ϵλ, λγ, τIminusXᵀX, XᵀYs[y]) for y in 1:length(data.ys)]
        end
        return βs
    end

    function l1l2(data::Data; λ::Float64 = 0.5, γ::Float64 = 1.0, ϵ::Float64 = 1.0,
                  ξ::Float64 = 0.001)::Vector{ManualModel}
        # λ, γ, ϵ, ξ = (0.5, 1.0, 1.0, 0.001)
        weights = weight_loop(data, λ, γ, ϵ, ξ)
        return ManualModel.(weights, true, data.ys, data)
    end

end

using .L1L2
export l1l2
