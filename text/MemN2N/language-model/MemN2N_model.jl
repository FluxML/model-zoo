import Flux: TrackedArray, param, onehot, OneHotVector, back!, relu, logitcrossentropy
import Flux.Optimise: Momentum, SGD
using BSON: @save
import NNlib: logsoftmax
#import JLD: save, load
using LinearAlgebra
struct Data
    train::Array
    num_words::Int
end

mutable struct Memory
    """
    Holds the weight matrices for memory and time
    """
    edim::Int64
    lindim::Int64
    nhops::Int64
    mem_size::Int64
    init_hid::Float64
    A::TrackedArray
    C::TrackedArray
    T_A::TrackedArray
    T_C::TrackedArray
    W::TrackedArray

end

function create_memory(nwords::Int64, edim::Int64,lindim::Int64, mem_size::Int64, init_hid::Float64, init_std::Float64, nhops::Int64)
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
    C = param(randn(nwords, edim)*init_std)

    T_A = param(randn(mem_size, edim)*init_std)
    T_C = param(randn(mem_size, edim)*init_std)

    W = param(randn(edim, nwords)*init_std)

    return Memory(edim, lindim, nhops, mem_size, init_hid, A, C, T_A, T_C, W)
end

function run_one_input(memory::Memory, input::Array, time::Array, context::Array)
    """
    Run an input through the memory and return final hidden layer
    Arguments:
    memory::Memory: holds the memory of network
    input::Array: array of indices of words
    time::Array: holds time for each element in context
    context::Array: the text corpus that will form the context for the current run
    """
    hidden = [param(input)]
    Ain_c = memory.A[context,:]
    Ain_t = memory.T_A[time,:]
    Ain = Ain_c + Ain_t

    Cin_c = memory.C[context,:]
    Cin_t = memory.T_C[time,:]
    Cin = Cin_c + Cin_t

    for hop=1:memory.nhops
        hid3dim = hidden[end]
        Aout = hid3dim*Ain'
        P = reshape(exp.(logsoftmax(Aout[1:end])), (1,:))
        Cout = P*Cin
        u_k = Cout +  hidden[end]
        if memory.lindim == memory.edim
            push!(hidden, u_k)
        elseif memory.lindim==0
            push!(hidden, relu.(u_k))
        else

            u_k_lin = u_k[1, 1:memory.lindim]
            u_k_nonLin = relu.(u_k[1, memory.lindim+1:end])
            u_k_combined = reshape(cat(u_k_lin, u_k_nonLin, dims=1), (1,:))
            push!(hidden, u_k_combined)
        end
    end
    return hidden[end]
end

function run_batch(memory::Memory, input::Array, time::Array, context::Array, targets::Array)
    """
    Runs the function run_one_input on a batch of inputs and returns the probabilities over words
        """
    z_s = []
    batch_size = size(input)[1]
    batch_cost = 0
    for i=1:batch_size
        input_i = reshape(input[i,:], (1, memory.edim))
        final_hidden = run_one_input(memory, input_i, time[i,:], context[i, :])
        if any(isnan, final_hidden)
            println(i)
        end
        z = dropdims(final_hidden*memory.W,dims=1)
        c = logitcrossentropy(z, targets[i,:])
        batch_cost += c
    end
    return batch_cost
end



function loss(logits::TrackedArray, targets::Array)
    """
    calculates crossentropy loss for each sample and returns a sum
    """
    costs = []
    batch_size = size(logits)[1]
    for i=1:batch_size
        push!(costs, logitcrossentropy(logits[i,:], targets[i,:]))
    end
    return cat(1, costs...)
end
#add train, optimizer, gradient clipping

function get_trainables(memory::Memory)
    """
    Returns trainable parameters of memory
    """
    trainables = []
    for fieldname in fieldnames(typeof(memory))
        field = getfield(memory, fieldname)
        if isa(field, TrackedArray)
            push!(trainables, field)
        end
    end
    return trainables
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


function create_training_batch(data, num_words, batchsize, memory::Memory)

    x = fill(memory.init_hid, (batchsize, memory.edim))
    time=zeros(Int64, (batchsize, memory.mem_size))
    for t=1:memory.mem_size
        time[:,t].=t
    end

    target = zeros(Int64, (batchsize, num_words))
    context = ones(Int64, (batchsize, memory.mem_size))

    for b=1:batchsize
        target_index = rand(memory.mem_size+1:length(data),1)[1]
        target[b, data[target_index]] = 1
        context[b,:] = data[target_index-memory.mem_size:target_index-1]
    end
    return x, target, time, context
end


function train(data::Data, memory::Memory, epochs::Int, batchsize::Int, max_norm::Int, η::Float64)
    """

    Run the model on data and perform gradient descent
    """
    trainables = get_trainables(memory)
    opt = SGD(trainables, η)

    N = convert(Int, ceil(length(data.train)/batchsize))
    cost = 0
    for epoch=1:epochs
        for i=1:N
            x, targets, time, context = create_training_batch(data.train, data.num_words, batchsize, memory)
            cost = run_batch(memory, x, time, context,targets)
            back!(cost)
            clip_gradients(trainables, max_norm)
            opt()
            @show i, cost
            if i%200==0
                @save "model_$(i)_$(round(Int, cost.data))" memory
            end
        end
    end
end


#TO DO: Inference
