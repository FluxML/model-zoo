using Flux
using Flux: crossentropy, throttle
using Flux.Data: Tree, children, isleaf

include("data.jl")

N = 300

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
    c1, l1 = forward(tree[1])
    c2, l2 = forward(tree[2])
    phrase = combine(c1, c2)
    phrase, l1 + l2 + crossentropy(sentiment(phrase), sent)
  end
end

loss(tree) = forward(tree)[2]

opt = ADAM()
ps = params(embedding, W, sentiment)
evalcb = () -> @show loss(train[1])

Flux.train!(loss, ps, zip(train), opt,
           cb = throttle(evalcb, 10))
