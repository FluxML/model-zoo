using Flux
using Flux: onehotbatch, unstack, truncate!, throttle

cd(@__DIR__)

isfile("input.txt") ||
  download("http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
           "input.txt")

text = collect(readstring("input.txt"))
alphabet = unique(text)

batchseq(xs, n) = unstack(reshape(xs[1:length(xs)÷n*n], :, n), 1)
subseq(xs, n) = [Seq(x) for x in Iterators.partition(xs, n)]

N = length(alphabet)
nseq = 50
nbatch = 50

Xs = subseq(map(b -> onehotbatch(b, alphabet), batchseq(text, nbatch)), nseq)
Ys = subseq(map(b -> onehotbatch(b, alphabet), batchseq(text[2:end], nbatch)), nseq)

m = ChainSeq(
  LSTM(N, 256),
  Dense(256, N),
  softmax)

seqloss(f, xs, ys) = sum(f(x, y) for (x, y) in zip(xs.data, ys.data))

loss(xs, ys) = seqloss(Flux.logloss, m(xs), ys)

evalcb = () -> @show loss(Xs[5], Ys[5])

Flux.train!(loss, zip(Xs, Ys), SGD(params(m), 1e-3),
            cb = [() -> truncate!(m),
                  throttle(evalcb, 2)])
