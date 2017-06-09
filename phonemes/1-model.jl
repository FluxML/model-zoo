# Based on https://arxiv.org/abs/1409.0473

using Flux: param, logloss, flip, stateless, broadcastto, ∘

include("0-data.jl")

Nin = length(alphabet)
Nhidden = 30

# A recurrent model which takes a token and returns a context-dependent
# annotation.

forward  = LSTM(Nin, Nhidden÷2)
backward = flip(LSTM(Nin, Nhidden÷2))
encoder  = @net token -> hcat(forward(token), backward(token))

alignnet = Affine(2Nhidden, 1)
align  = @net (s, t) -> alignnet(hcat(broadcastto(s, (Nbatch, 1)), t))

# A recurrent model which takes a sequence of annotations, attends, and returns
# a predicted output token.

recur   = unroll1(LSTM(Nhidden, Nhidden)).model
state   = param(zeros(1, Nhidden))
y       = param(zeros(1, Nhidden))
toalpha = Affine(Nhidden, length(phones))

decoder = @net function (tokens)
  energies = map(token -> exp.(align(state{-1}, token)), tokens)
  weights = map(e -> e ./ sum(energies), energies)
  context = sum(map(∘, weights, tokens))
  (y, state), _ = recur((y{-1},state{-1}), context)
  return softmax(toalpha(y))
end

# Building the full model

encoder, decoder = open(deserialize, "model.jls")

model = @Chain(
  stateless(unroll(encoder, Nseq)),
  @net(x -> repeated(x, Nseq)),
  stateless(unroll(decoder, Nseq)))

mxmodel = mxnet(Flux.SeqModel(model, Nseq))

evalcb = () -> @show logloss(rawbatch(mxmodel(first(Xs))), rawbatch(first(Ys)))

# @time Flux.train!(mxmodel, zip(Xs, Ys), η = 1e-3,
#             loss = Flux.logloss,
#             cb = [evalcb])

# open(io -> serialize(io, (encoder, decoder)), "model.jls", "w")
