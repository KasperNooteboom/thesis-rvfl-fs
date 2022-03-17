module MLPMod

using Flux, Flux.Optimise, Flux.Losses
using StatsBase, StatsModels
using ..DataMod, ..Helpers

export MLPModel
export MLP, trainMLP

struct MLPModel <: RegressionModel
    mlp
    data::Data
    data_test::Data
    best_epoch::Int
    total_epochs::Int
    params::Vector
    loss::Float64
    stopped::Symbol
end

Base.display(m::MLPModel) = Base.display(m.mlp)
StatsBase.predict(m::MLPModel, A::Matrix{Float64}) = Matrix(size(params(m.mlp)[1], 2) == size(A, 1) ? m.mlp(A)' : m.mlp(A')')
StatsBase.predict(m::MLPModel) = Matrix(m.mlp(m.data.xmat')')

function MLP(nodes...)
    length(nodes) < 2 && throw("MLP needs at least an input size and an output size.")
    if nodes[end] > 1
        Chain([[Dense(nodes[i], nodes[i+1], σ) for i in eachindex(nodes)[1:(end-2)]]; Dense(nodes[end-1], nodes[end]); softmax]...)
    else
        Chain([[Dense(nodes[i], nodes[i+1], σ) for i in eachindex(nodes)[1:(end-2)]]; Dense(nodes[end-1], nodes[end])]...)
    end
end
MLP(nodes::Vector{Int}) = MLP(nodes...)

function progress(striploss::Vector{Float64})
    k = length(striploss)
    num = sum(striploss)
    denom = k * minimum(striploss)
    return 1000 * (num / denom - 1)
end

function trainMLP(data::Data, data_test::Data; s::Int = 4, k::Int = 5,
                  stop_progress::Float64 = 0.1, max_epoch::Int = 3000, verbose::Bool = true)::MLPModel
    train = zip(eachrow(data.xmat), eachrow(data.ymat))
    nin = length(data.xs)
    nout = length(data.ys)
    h1 = min(2 * nin, 100)
    h2 = min(floor(Int, 1.3 * nin), 65)
    mlp = MLP(nin, h1, h2, nout)
    if verbose
        println("architecture of MLP:")
        display(mlp)
    end
    loss(x, y) = mse(mlp(x), y)
    ps = params(mlp)
    xtrn, ytrn = data.xmat', data.ymat'
    xtst, ytst = data_test.xmat', data_test.ymat'
    ltrn, ltst = loss(xtrn, ytrn), loss(xtst, ytst)
    pltrn, pltst = xdecim(ltrn, 5), xdecim(ltst, 5)
    verbose && println("[Before training] ~$pltrn training loss; ~$pltst testing loss")
    lastloss = Inf; csc_gain = 0; striploss = Float64[]
    bestloss = Inf; bestprms = []; bestepoch = 0
    total_epochs = 0
    stopped = :maxiter
    for epoch in 1:max_epoch
        total_epochs = epoch
        verbose && print("[Epoch $epoch] ")
        train!(loss, ps, train, ADAM())
        ltrn, ltst = loss(xtrn, ytrn), loss(xtst, ytst)
        pltrn, pltst = xdecim(ltrn, 5), xdecim(ltst, 5)
        verbose && println("~$pltrn training loss; ~$pltst testing loss")
        if ltst < bestloss
            bestprms = collect(params(cpu(mlp)))
            bestloss = ltst
            bestepoch = epoch
        end
        push!(striploss, ltrn)
        if epoch % k == 0
            if ltst > lastloss
                csc_gain += 1
                if csc_gain >= s
                    verbose && println("-- testing loss increased in $s consecutive strips")
                    stopped = :consecincrease
                    break
                elseif verbose
                    println("-- testing loss increased this strip")
                end
            else
                csc_gain = 0
            end
            lastloss = ltst
            if progress(striploss) < stop_progress
                verbose && println("-- progress this strip was < $stop_progress")
                stopped = :noprogress
                break
            end
            empty!(striploss)
        end
    end
    verbose && println("returning model with lowest testing loss ($bestloss) found on epoch $bestepoch")
    Flux.loadparams!(mlp, bestprms)
    return MLPModel(mlp, data, data_test, bestepoch, total_epochs, bestprms, bestloss, stopped)
end

end

using .MLPMod