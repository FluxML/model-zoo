# Based on https://arxiv.org/abs/1409.0473
include("0-data.jl")
using Flux: flip, logitcrossentropy, reset!, throttle
using Parameters: @with_kw
using StatsBase: wsample

@with_kw mutable struct Args
    lr::Float64 = 1e-3      # learning rate
    Nin::Int = 0            # size of input layer, will be assigned as length(alphabet)
    Nh::Int = 30            # size of hidden layer
    phones_len::Int = 0     # length of phonemes
    throttle::Int = 30      # throttle timeout
end

function build_model(args)
    # A recurrent model which takes a token and returns a context-dependent
    # annotation.
    forward  = LSTM(args.Nin, args.Nh÷2)
    backward = LSTM(args.Nin, args.Nh÷2)
    encode(tokens) = vcat.(forward.(tokens), flip(backward, tokens))

    alignnet = Dense(2*args.Nh, 1)

    # A recurrent model which takes a sequence of annotations, attends, and returns
    # a predicted output token.
    recur   = LSTM(args.Nh+args.phones_len, args.Nh)
    toalpha = Dense(args.Nh, args.phones_len)
    return (forward, backward, alignnet, recur, toalpha), encode
end

align(s, t, alignnet) = alignnet(vcat(t, s .* Int.(ones(1, size(t, 2)))))

function asoftmax(xs)
  xs = [exp.(x) for x in xs]
  s = sum(xs)
  return [x ./ s for x in xs]
end

function decode1(tokens, phone, state)
    # Unpack models
    forward, backward, alignnet, recur, toalpha = state
    weights = asoftmax([align(recur.state[2], t, alignnet) for t in tokens])
    context = sum(map((a, b) -> a .* b, weights, tokens))
    y = recur(vcat(Float32.(phone), context))
    return toalpha(y)
end

decode(tokens, phones, state) = [decode1(tokens, phone, state) for phone in phones]

function model(x, y, state, encode)
    # Unpack models
    forward, backward, alignnet, recur, toalpha = state
    ŷ = decode(encode(x), y, state)
    reset!(state)
    return ŷ
end

function predict(s, state, encode, alphabet, phones)
    ts = encode(tokenise(s, alphabet))
    ps = Any[:start]
    for i = 1:50
      dist = softmax(decode1(ts, onehot(ps[end], phones), state))
      next = wsample(phones, vec(dist))
      next == :end && break
      push!(ps, next)
    end
    reset!(state)
    return ps[2:end]
end

function train(; kws...)
    # Initialize Hyperparameters
    args = Args(; kws...)
    @info("Loading Data...")
    data,alphabet,phones = getdata(args)

    # The full model
    # state = (forward, backward, alignnet, recur, toalpha)
    @info("Constructing Model...")
    state, encode = build_model(args)

    loss(x, yo, y) = sum(logitcrossentropy.(model(x, yo, state, encode), y))
    evalcb = () -> @show loss(data[500]...)
    opt = ADAM(args.lr)
    @info("Training...")
    Flux.train!(loss, params(state), data, opt, cb = throttle(evalcb, args.throttle))
    return state, encode, alphabet, phones
end

cd(@__DIR__)
state, encode, alphabet, phones = train()
@info("Testing...")
predict("PHYLOGENY", state, encode, alphabet, phones)
