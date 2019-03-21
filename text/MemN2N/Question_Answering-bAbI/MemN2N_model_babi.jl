import Flux: TrackedArray, param, onehot, OneHotVector, back!, relu, logitcrossentropy
import Flux.Optimise: Momentum, SGD
using BSON: @save
import NNlib: logsoftmax
#import JLD: save, load
using LinearAlgebra
using Random
using JLD

function position_encoding(sentence_size, embedding_size)
    encoding = ones(Float32, embedding_size, sentence_size)
    for i=1:embedding_size
        for j=1:sentence_size
            encoding[i,j] .= (i - (embedding_size+1)/2)* (j - (sentence_size+1)/2)
        end
    end
    encoding = 1 .+ 4 .* ((encoding ./ embedding_size) ./ sentence_size)
    encoding[:, end] .= 1.0
    return encoding'
end

#const p_encode = position_encoding(10, 20)
# const sentence_size=10
# hopn=3
mutable struct Memory
    """
    Holds the weight matrices for memory and time
    """
    nwords::Int64
    edim::Int64
    lindim::Int64
    nhops::Int64
    mem_size::Int64
    init_hid::Float64
    A::TrackedArray
    C::Array
    T_A::TrackedArray
    T_C::TrackedArray
    p_encode::Array
end

function create_memory(nwords::Int64, edim::Int64,lindim::Int64, mem_size::Int64, init_hid::Float64, init_std::Float64, nhops::Int64, sentence_size::Int64)
    """
    Creates an instance of memory
    Arguments
    nwords::Int64: number of words in vocabulary
    edim::Int64: embedding dimension for words
    mem_size::Int64: size of memory
    init_std::Float64: standard deviation to initialize weights
    nhops::Int64: number of hops through memory before output
    """
    A = param(randn(nwords, edim)*init_std)
    C=[]

    for i=1:nhops
        push!(C, param(randn(nwords, edim)*init_std))
    end

    T_A = param(randn(mem_size, edim)*init_std)
    T_C = param(randn(mem_size, edim)*init_std)

    p_encode = position_encoding(sentence_size, edim)

    return Memory(nwords, edim, lindim, nhops, mem_size, init_hid, A, C, T_A, T_C, p_encode)
end

function run_one_input(memory::Memory, stories::Array,query::Array, time::Array)
    """
    Run an input through the memory and return final hidden layer
    Arguments:
    memory::Memory: holds the memory of network
    input::Array: array of indices of words
    time::Array: holds time for each element in context
    context::Array: the text corpus that will form the context for the current run
    """

    q_emb = memory.A[query,:]
    u_0 = sum(q_emb .* memory.p_encode, dims=1)
    u = [u_0]


    for i_hop in 1:memory.nhops
        m_emb_A = []
        m_emb_C = []
        if i_hop == 1
            for i_story=1:length(stories)
                push!(m_emb_A, cat(memory.A[stories[i_story],:].*memory.p_encode, reshape(memory.T_A[i_story,:],(1,memory.edim)), dims=1))

            end
            m_emb_A_sum =[]
            for i in m_emb_A
                push!(m_emb_A_sum, sum(i,dims=1))
            end
            m = vcat(m_emb_A_sum...)

        else
            for i_story=1:length(stories)
                push!(m_emb_C, cat(memory.C[i_hop-1][stories[i_story],:].*memory.p_encode, reshape(memory.T_C[i_story,:],(1,memory.edim)), dims=1))
            end
            m_emb_C_sum =[]
            for i in m_emb_C
                push!(m_emb_C_sum, sum(i,dims=1))
            end
            m = vcat(m_emb_C_sum...)
        end

        dotted = sum(m .* u[end], dims=2)
        probs = exp.(logsoftmax(dotted))

        m_emb_C = []
        for i_story=1:length(stories)
            push!(m_emb_C, memory.C[i_hop][stories[i_story],:].*memory.p_encode)
        end
        m_emb_C_sum =[]
        for i in m_emb_C
            push!(m_emb_C_sum, sum(i,dims=1))
        end
        m_C = vcat(m_emb_C_sum...)


        o_k = sum(m_C .* probs, dims=1)
        u_k = u[end] .+ o_k
        push!(u, u_k)
    end
    out = reshape((u[end])*memory.C[end]', (memory.nwords,))
    return out
end

function run_batch(memory::Memory, data::Array)
    batch_cost = 0
    for (story_vec, query_vec, answer_vec, time_vec) in data
        logits = run_one_input(memory, story_vec, query_vec, time_vec)

        batch_cost += logitcrossentropy(logits, answer_vec)
    end
    return batch_cost
end


function clip_gradients(trainables, max_norm)
    """
    clips gradients to max norm
    """
    for trainable in trainables
        grad_val = trainable.grad
        grad_norm = norm(grad_val,2)
        # println(grad_norm)
        if grad_norm<=max_norm
            continue
        else
            grad_val = (grad_val/grad_norm)*max_norm
            for i=1:length(grad_val)
                trainable.grad[i] = grad_val[i]
            end
        end
    end
end

function zero_nil_slot(memory::Memory)
    memory.A.grad[1,:].=0.0
    for i in memory.C
        i.grad[1,:].=0.0
    end
end
function train(data::Array, memory::Memory, epochs::Int, batchsize::Int, max_norm::Int, η::Float64)
    """

    Run the model on data and perform gradient descent
    """
    f = open("cost.txt", "w")
    trainables = [memory.A, memory.T_A, memory.T_C]
    for i in memory.C
        push!(trainables, i)
    end
    opt = SGD(trainables, η)
    cost = 0
    for epoch=1:epochs
        # JLD.save("memory"*"$(epoch)_$(η)"*".jld", "mem", memory)
        if epoch%15==0
            η = η/2
            opt = SGD(trainables, η)
        end

        shuffle!(data)
        for i=1:batchsize:length(data)-batchsize
            cost = run_batch(memory, data[i:i+batchsize-1])
            back!(cost)
            if isnan(sum(memory.A.grad)) || isnan(sum(memory.T_A.grad))
                JLD.save("memory.jld", "mem", memory)
                JLD.save("data.jld", "val", data[i:i+batchsize-1])
            end
            write(f, string(i)*"   "*string(cost.data)*"  \n")
            clip_gradients(trainables, max_norm)
            zero_nil_slot(memory)
            opt()
            @show i, cost
        end
    end
end
