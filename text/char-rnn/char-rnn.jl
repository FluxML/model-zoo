using Flux
using Flux: onehot, chunk, batchseq, throttle, logitcrossentropy
using StatsBase: wsample
using Base.Iterators: partition
using Parameters: @with_kw

# Hyperparameter arguments 
@with_kw mutable struct Args
    lr::Float64 = 1e-2	# Learning rate
    seqlen::Int = 50	# Length of batchseqences
    nbatch::Int = 50	# number of batches text is divided into
    throttle::Int = 30	# Throttle timeout
end

function getdata(args)
    # Download the data if not downloaded as 'input.txt'
    isfile("input.txt") ||
        download("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt","input.txt")

    text = collect(String(read("input.txt")))
    
    # an array of all unique characters
    alphabet = [unique(text)..., '_']
    
    text = map(ch -> onehot(ch, alphabet), text)
    stop = onehot('_', alphabet)

    N = length(alphabet)
    
    # Partitioning the data as sequence of batches, which are then collected as array of batches
    Xs = collect(partition(batchseq(chunk(text, args.nbatch), stop), args.seqlen))
    Ys = collect(partition(batchseq(chunk(text[2:end], args.nbatch), stop), args.seqlen))

    return Xs, Ys, N, alphabet
end

# Function to construct model
function build_model(N)
    return Chain(
            LSTM(N, 128),
            LSTM(128, 128),
            Dense(128, N))
end 

function train(; kws...)
    # Initialize the parameters
    args = Args(; kws...)
    
    # Get Data
    Xs, Ys, N, alphabet = getdata(args)

    # Constructing Model
    m = build_model(N)

    function loss(xs, ys)
      l = sum(logitcrossentropy.(m.(xs), ys))
      return l
    end
    
    ## Training
    opt = ADAM(args.lr)
    tx, ty = (Xs[5], Ys[5])
    evalcb = () -> @show loss(tx, ty)

    Flux.train!(loss, params(m), zip(Xs, Ys), opt, cb = throttle(evalcb, args.throttle))
    return m, alphabet
end

# Sampling
function sample(m, alphabet, len; seed="")
    m = cpu(m)
    Flux.reset!(m)
    buf = IOBuffer()
    if seed == ""
        seed = string(rand(alphabet))
    end
    write(buf, seed)
    c = wsample(alphabet, softmax(m.(map(c -> onehot(c, alphabet), collect(seed)))[end]))
    for i = 1:len
        write(buf, c)
        c = wsample(alphabet, softmax(m(onehot(c, alphabet))))
    end
    return String(take!(buf))
end

cd(@__DIR__)
m, alphabet = train()
sample(m, alphabet, 1000) |> println
