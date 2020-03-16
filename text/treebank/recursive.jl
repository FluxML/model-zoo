using Flux
using Flux: crossentropy, throttle
using Flux.Data: Tree, children, isleaf
using Parameters: @with_kw
include("data.jl")

@with_kw mutable struct Args
    lr::Float64 = 1e-3    # Learning rate
    N::Int = 300
    throttle::Int = 10    # Throttle timeout
end

function train(; kws...)
    # Initialize HyperParameters
    args = Args(; kws...)
    # load data
    @info("Loading Data...")
    train_data = getData(args)    

    @info("Constructing model....")
    embedding = param(randn(Float32, N, length(alphabet)))

    W = Dense(2N, N, tanh)
    combine(a, b) = W([a; b])

    sentiment = Chain(Dense(N, 5), softmax)

    function forward(tree)
      if isleaf(tree)
        token, sent = tree.value
        phrase = embedding * token
        phrase, crossentropy(sentiment(phrase), sent)
      else
        _, sent = tree.value
        c1, l1 = forward(tree[1], embedding)
        c2, l2 = forward(tree[2], embedding)
        phrase = combine(c1, c2)
        phrase, l1 + l2 + crossentropy(sentiment(phrase), sent)
      end
    end

    loss(tree) = forward(tree)[2]
 
    opt = ADAM(args.lr)
    ps = params(embedding, W, sentiment)
    evalcb = () -> @show loss(train[1])
    @info("Training Model...")
    Flux.train!(loss, ps, zip(train_data), opt,cb = throttle(evalcb, args.throttle))
end
