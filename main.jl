using Pkg; Pkg.activate(".")
using JSON, Random

include("./incl/helpers.jl")
include("./incl/data.jl")
# include("./incl/mlp.jl")
include("./incl/RVFL.jl")
include("./incl/fs.jl")

function run()
    RVFL_RUNS = 30
    ALG_START, READ_START, RVFL_START, CV_START = 1, 1, 1, 1
    algs = [importance, importance_corfilter, stepwise, lasso, sparse_encoded_lasso, l1l2, inipg, ga_uniform]
    readers = readers_full()

    # Toy dataset is mushroom dataset with only the last 10 features and only the first 200 observations
    println("Reading toy dataset...")
    data = read_toy()
    rvfl = RVFL(length(data.xs), 2)
    edata = enhanced(rvfl, data)
    for algi in eachindex(algs)
        algi < ALG_START && continue
        alg = algs[algi]
        origalg = algs[algi]
        stralg = string(alg)
        if algi ∈ [3, 7, 8, 9]
            alg = d -> origalg(d, verbose = false)
        end
        println("Preparing algorithm $stralg by running it once on a toy dataset")
        alg(edata)
    end
    println("Done preparing. Running for real.")

    for algi in eachindex(algs)
        algi < ALG_START && continue
        firstround = algi == ALG_START
        alg = algs[algi]
        origalg = algs[algi]
        stralg = string(alg)
        println("Using algorithm $stralg")
        filename = "performance_mingood_$stralg"
        printheader = !firstround || (RVFL_START == 1 && CV_START == 1)
        printheader && log2file(filename, "run,dataset,nin,nout,time,ftrain,ftest,atrain,atest,nsel,nenhanced,sel,coefs")
        for ri in eachindex(readers)
            (firstround && ri < READ_START) && continue
            firstround &= ri == READ_START
            r = readers[ri]
            dataset = r.name
            println("Reading dataset $dataset...")
            alldata = r.cv ? r.read() : [r.read()]
            for i in 1:RVFL_RUNS
                (firstround && i < RVFL_START) && continue
                firstround &= i == RVFL_START
                if algi ∈ [7, 8]
                    alg = d -> origalg(d, MersenneTwister(100i), verbose = false)
                elseif algi == 3
                    alg = d -> origalg(d, verbose = false)
                end
                for cv in eachindex(alldata)
                    (firstround && cv < CV_START) && continue
                    firstround &= cv == CV_START
                    data, data_test = alldata[cv]
                    nout = length(data.ys)
                    nout == 1 && (nout += 1)
                    cvtxt = r.cv ? " - CV fold $cv" : ""
                    println("Algorithm $stralg - dataset $dataset ($nout classes) - RVFL run $i$cvtxt")
                    rvfl = RVFL(length(data.xs), r.m, MersenneTwister(i))
                    edata, edata_test = [enhanced(rvfl, d) for d in [data, data_test]]
                    time = @elapsed models = alg(edata)
                    ftrain, ftest = [f1(models, d) for d in [edata, edata_test]]
                    atrain, atest = [prediction_accuracy(models, d) for d in [edata, edata_test]]
                    selected = featurestrings(models)
                    nsel = JSON.json([sum(collect(s) .== '1') for s in selected])
                    nenhanced = JSON.json([sum(collect(s[(end - r.m + 1):end]) .== '1') for s in selected])
                    selected_json = '[' * join(selected, ',') * ']'
                    coefs = JSON.json(get_coefs(models))
                    nin = length(edata.xs)
                    log2file(filename, "$i,$dataset,$nin,$nout,$time,$ftrain,$ftest,$atrain,$atest,\"$nsel\",\"$nenhanced\",\"$selected_json\",\"$coefs\"")
                end
            end
        end
    end
end
run()
