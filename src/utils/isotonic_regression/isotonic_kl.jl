function isotonic_kl(s::AbstractVector, w::AbstractVector)
    n = length(s)
    target = [i for i in 1:n] # if block i -> j, then target[i] = j and target[j] = i
    logsumexp_s = zeros(n) .+ s
    logsumexp_w = zeros(n) .+ w
    sol = logsumexp_s .- logsumexp_w

    i = 1
    while i <= n
        k = target[i] + 1 # start of next block
        if k == n + 1
            break
        end
        if sol[i] > sol[k]  # continue if B and B+ are correctly ordered
            i = k
            continue
        end

        # merge B with B+
        logsumexp_s[i] = logaddexp(logsumexp_s[i], logsumexp_s[k])
        logsumexp_w[i] = logaddexp(logsumexp_w[i], logsumexp_w[k])

        sol[i] = logsumexp_s[i] - logsumexp_w[i]

        k = target[k] + 1  # start of next block
        target[i] = k - 1
        target[k - 1] = i

        if i > 1
            # check if we now need to merge some blocks before i
            i = target[i - 1]
        end
    end

    # reconstruct solution
    i = 1
    while i <= n
        k = target[i] + 1
        sol[(i + 1):(k - 1)] .= sol[i]
        i = k
    end
    return sol
end

function isotonic_kl_with_sizes(s::AbstractVector, w::AbstractVector)
    n = length(s)
    target = [i for i in 1:n] # if block i -> j, then target[i] = j and target[j] = i
    logsumexp_s = zeros(n) .+ s
    logsumexp_w = zeros(n) .+ w
    sol = logsumexp_s .- logsumexp_w

    nb_blocks = n  # keep track of number of clock

    i = 1
    while i <= n
        k = target[i] + 1 # start of next block
        if k == n + 1
            break
        end
        if sol[i] > sol[k]  # continue if B and B+ are correctly ordered
            i = k
            continue
        end

        # merge B with B+
        logsumexp_s[i] = logaddexp(logsumexp_s[i], logsumexp_s[k])
        logsumexp_w[i] = logaddexp(logsumexp_w[i], logsumexp_w[k])

        sol[i] = logsumexp_s[i] - logsumexp_w[i]

        k = target[k] + 1  # start of next block
        target[i] = k - 1
        target[k - 1] = i
        nb_blocks -= 1

        if i > 1
            # check if we now need to merge some blocks before i
            i = target[i - 1]
        end
    end

    # reconstruct solution
    i = 1
    while i <= n
        k = target[i] + 1
        sol[(i + 1):(k - 1)] .= sol[i]
        i = k
    end

    block_sizes = zeros(Int, nb_blocks)
    current_index = 1
    for i in eachindex(block_sizes)
        width = 1
        value = sol[current_index]
        while current_index < n && isapprox(value, sol[current_index + 1]; atol=1e-9)
            current_index += 1
            width += 1
        end
        current_index += 1
        block_sizes[i] = width
    end

    return sol, block_sizes
end

function ChainRulesCore.rrule(::typeof(isotonic_kl), s::AbstractVector, w::AbstractVector)
    ŷ, sizes = isotonic_kl_with_sizes(s, w)

    # # TODO: probably can do better (without push! allocations)
    # widths = [1]
    # for i in eachindex(ŷ)
    #     if i == length(s)
    #         break
    #     end
    #     if !isapprox(ŷ[i], ŷ[i + 1]; atol=1e-9)
    #         push!(widths, 0)
    #     end
    #     widths[end] += 1
    # end

    function isotonic_pullback(Δy)
        res_s = zeros(length(Δy))
        res_w = zeros(length(Δy))
        start = 1
        for width in sizes
            slice = start:(start + width - 1)
            # @show ŷ widths slice softmax(s[slice]) Δy[slice] softmax(w[slice])
            println()
            res_s[slice] .= sum(Δy[slice]) .* softmax(s[slice])
            res_w[slice] .= -sum(Δy[slice]) .* softmax(w[slice])
            start += width
        end
        return NoTangent(), res_s, res_w
    end

    return ŷ, isotonic_pullback
end

# ! pullback using forawrd autodiff
# function isotonic_pullback(Δy)
#     res_s = zeros(length(Δy))
#     res_w = zeros(length(Δy))
#     start = 1
#     for width in widths
#         slice = start:(start + width - 1)
#         @show ŷ widths slice softmax(s[slice]) Δy[slice] softmax(w[slice])
#         println()
#         res_s[slice] .= dot(softmax(s[slice]), Δy[slice])
#         res_w[slice] .= -dot(softmax(w[slice]), Δy[slice])
#         start += width
#     end
#     @show res_w
#     return NoTangent(), res_s, res_w
# end
