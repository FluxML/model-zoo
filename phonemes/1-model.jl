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

recur   = unroll1(LSTM(Nhidden+length(phones), Nhidden)).model
state   = param(zeros(1, Nhidden))
y       = param(zeros(1, Nhidden))
toalpha = Affine(Nhidden, length(phones))

decoder = @net function (tokens, phone)
  energies = map(token -> exp.(align(state{-1}, token)), tokens)
  weights = map(e -> e ./ sum(energies), energies)
  context = sum(map(∘, weights, tokens))
  (y, state), _ = recur((y{-1},state{-1}), hcat(phone, context))
  return softmax(toalpha(y))
end

# Building the full model

# encoder, decoder = open(deserialize, "model.jls")

encoderu = stateless(unroll(encoder, Nseq))
decoderu = stateless(unroll(decoder, Nseq))

model = @net function (input)
  x, y = input
  tokens = encoderu(x)
  decoderu(repeated(tokens, Nseq), y)
end

mxmodel = mxnet(Flux.SeqModel(model, Nseq))

mxmodel((first(Xs), first(Yoffset)))

evalcb = () -> @show logloss(rawbatch(mxmodel((first(Xs), first(Yoffset)))), rawbatch(first(Ys)))

# @time Flux.train!(mxmodel, zip(zip(Xs, Yoffset), Ys), η = 1e-3,
#                   loss = logloss, cb = [evalcb])

# open(io -> serialize(io, (encoder, decoder)), "model.jls", "w")
