nanmean(x::AbstractVector)::Float64 = mean(map(n -> isnan(n) ? 0 : n, x))
nanmean(x::AbstractMatrix)::Vector{Float64} = map(nanmean, eachrow(x))
absnanmean(x::AbstractVector)::Float64 = nanmean(map(abs, x))
absnanmean(x::AbstractMatrix)::Vector{Float64} = map(absnanmean, eachrow(x))

# The code for `calc_drop` and `drop_cor` below is inspired by the code by Brian Pietracatella, found at
# https://towardsdatascience.com/are-you-dropping-too-many-correlated-features-d1c96654abe6

function calc_drop(res::DataFrame)::Vector{Symbol}
    # Get all variables that have some high correlation
    all_corr_vars = unique([res[!, :v1]; res[!, :v2]])
    # Get all variables that are possibly dropped
    poss_drop = unique(res[!, :drop])
    # Get variables that are definitely kept
    keep = setdiff(all_corr_vars, poss_drop)
    # Drop variables that are highly correlated to a variable that is definitely kept
    p = res[in(keep).(res[!, :v1]) .| in(keep).(res[!, :v2]), [:v1, :v2]]
    q = unique([p[!, :v1]; p[!, :v2]])
    drop = setdiff(q, keep)
    # Remove dropped variables from the list of possibilities
    poss_drop = setdiff(poss_drop, drop)
    # Get variables that (correlate highly to variables that) are still possibly dropped
    m = res[in(poss_drop).(res[!, :v1]) .| in(poss_drop).(res[!, :v2]), [:v1, :v2, :drop]]
    # Drop variables that are not (correlated to variables that are) already dropped
    more_drop = m[.!in(drop).(m[!, :v1]) .& .!in(drop).(m[!, :v2]), :drop]
    return [drop; more_drop]
end

function drop_cor(cormat::Matrix{Float64}, xs::Vector{Symbol}, t::Float64)::Vector{Symbol}
    msize = size(cormat, 1)
    (msize == size(cormat, 2)) || throw("cormat is not a square matrix")
    (msize == length(xs)) || throw("length of xs does not match size of cormat")

    avgcor = absnanmean(cormat)

    res = DataFrame(v1 = Symbol[], v2 = Symbol[], v1mean = Float64[], v2mean = Float64[], corr = Float64[], drop = Symbol[])

    @inbounds for row in 1:(msize - 1)
        for col in (row + 1):msize
            if abs(cormat[row, col]) > t
                rowsym = xs[row]
                colsym = xs[col]
                drop = avgcor[row] > avgcor[col] ? rowsym : colsym
                entry = [
                    rowsym, colsym,
                    avgcor[row], avgcor[col],
                    cormat[row, col], drop
                ]
                push!(res, entry)
            end
        end
    end

    return calc_drop(res)

end
