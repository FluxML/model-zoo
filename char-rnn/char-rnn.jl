using Flux
using Flux: onehotbatch, unstack, truncate!, throttle, logloss
using Base.Iterators: partition

cd(@__DIR__)

isfile("input.txt") ||
  download("http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
           "input.txt")

text = collect(readstring("input.txt"))
alphabet = unique(text)

batchseq(xs, n) = unstack(reshape(xs[1:length(xs)÷n*n], :, n), 1)

N = length(alphabet)
nseq = 50
nbatch = 50

Xs = partition(map(b -> onehotbatch(b, alphabet), batchseq(text, nbatch)), nseq) |> collect
Ys = partition(map(b -> onehotbatch(b, alphabet), batchseq(text[2:end], nbatch)), nseq) |> collect

m = Chain(
  LSTM(N, 256),
  Dense(256, N),
  softmax)

loss(xs, ys) = sum(logloss.(m.(xs), ys))

evalcb = () -> @show loss(Xs[5], Ys[5])

Flux.train!(loss, zip(Xs, Ys), SGD(params(m), 0.1),
            cb = [() -> truncate!(m),
                  throttle(evalcb, 2)])
