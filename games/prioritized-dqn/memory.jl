include("sum-tree.jl")

using Distributions

mutable struct Memory  # stored as ( s, a, r, s_ ) in SumTree
    ϵ::Float32  # small amount to avoid zero priority
    α::Float32  # [0~1] convert the importance of TD error to priority
    β::Float32  # importance-sampling, from initial value increasing to 1
    β_increment_per_sampling::Float32
    abs_err_upper::Float32  # clipped abs error
    tree::SumTree

    function Memory(capacity::Int, state_size::Int)
        ϵ = 0.01
        α = 0.6
        β = 0.4
        β_increment_per_sampling = 0.001
        abs_err_upper = 1.0
        tree = SumTree(capacity, state_size)

        new(ϵ, α, β, β_increment_per_sampling, abs_err_upper, tree)
    end
end

function store!(mem::Memory, transition)
    max_p = maximum(mem.tree.tree[end - mem.tree.capacity + 1:end])
    max_p = max_p == 0 ? mem.abs_err_upper : max_p

    add!(mem.tree, max_p, transition)  # set the max p for new p
end

function mem_sample(mem::Memory, n)
    b_idx = Array{Int32, 1}(n)
    b_memory = Array{Float32, 2}(length(mem.tree.data[:, 1]), n)
    ISWeights = Array{Float64, 2}(1, n)

    pri_seg = total_p(mem.tree) / n       # priority segment
    mem.β = min(1., mem.β + mem.β_increment_per_sampling)  # max = 1
    min_prob = minimum(mem.tree.tree[end-mem.tree.capacity+1:end]) / total_p(mem.tree)# for later calculate ISweight

    for i = 0:n-1
        a, b = pri_seg * i, pri_seg * (i + 1)
        v = rand(Uniform(a, b))
        idx, p, data = get_leaf(mem.tree, v)
        prob = p / total_p(mem.tree)
        ISWeights[1, i + 1] = float(min_prob / prob) ^ mem.β
        b_idx[i + 1], b_memory[:, i + 1] = idx, data
    end

    return b_idx, b_memory, ISWeights
end

function batch_update!(mem::Memory, tree_idx, abs_errors)
    abs_errors += mem.ϵ  # convert to abs and avoid 0
    clipped_errors = min.(abs_errors, mem.abs_err_upper)
    ps = clipped_errors .^ mem.α
    for (ti, p) in zip(tree_idx, ps)
        ti = convert(Int64, ti)
        update!(mem.tree, ti , p)
    end
end
