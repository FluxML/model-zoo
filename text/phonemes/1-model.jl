# Based on https://arxiv.org/abs/1409.0473

using Flux: flip, crossentropy, reset!, throttle, glorot_uniform

include("0-data.jl")

Nin = length(alphabet)
Nh = 30 # size of hidden layer

# A recurrent model which takes a token and returns a context-dependent
# annotation.

forward  = LSTM(Nin, Nh÷2)
backward = LSTM(Nin, Nh÷2)
encode(tokens) = vcat.(forward.(tokens), flip(backward, tokens))

Ws = param(glorot_uniform(1, Nh))
Wt = param(glorot_uniform(1, Nh))
b = param(zeros(1))
align(s,t) = Ws*s .+ Wt*t .+ b
    
# A recurrent model which takes a sequence of annotations, attends, and returns
# a predicted output token.

recur   = LSTM(Nh+length(phones), Nh)
toalpha = Dense(Nh, length(phones))

function asoftmax(xs)
  xs = [exp.(x) for x in xs]
  s = sum(xs)
  return [x ./ s for x in xs]
end

function decode1(tokens, phone)
  weights = asoftmax([align(recur.state[2], t) for t in tokens])
  context = sum(map((a, b) -> a .* b, weights, tokens))
  y = recur(vcat(Int32.(phone), context))
  return softmax(toalpha(y))
end

decode(tokens, phones) = [decode1(tokens, phone) for phone in phones]

# The full model

state = (forward, backward, recur, toalpha)

function model(x, y)
  ŷ = decode(encode(x), y)
  reset!(state)
  return ŷ
end

loss(x, yo, y) = sum(crossentropy.(model(x, yo), y))

evalcb = () -> @show loss(data[500]...)
opt = ADAM(params(state))

Flux.train!(loss, data, opt, cb = throttle(evalcb, 10))

# Prediction

using StatsBase: wsample

function predict(s)
  ts = encode(tokenise(s, alphabet))
  ps = Any[:start]
  for i = 1:50
    dist = decode1(ts, onehot(ps[end], phones))
    next = wsample(phones, dist.data)
    next == :end && break
    push!(ps, next)
  end
  return ps[2:end]
end

predict("PHYLOGENY")
