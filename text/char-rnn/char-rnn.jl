# uncomment to run on gpu, if available
#using CuArrays

using Flux
using Flux: onehot, argmax, chunk, batchseq, throttle, crossentropy
using StatsBase: wsample
using Base.Iterators: partition

cd(@__DIR__)

isfile("input.txt") ||
  download("http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
           "input.txt")

text = collect(readstring("input.txt"))
alphabet = [unique(text)..., '_']
text = map(ch -> onehot(ch, alphabet), text)
stop = onehot('_', alphabet)

N = length(alphabet)
seqlen = 50
nbatch = 50

Xs = collect(partition(batchseq(chunk(text, nbatch), stop), seqlen))
Ys = collect(partition(batchseq(chunk(text[2:end], nbatch), stop), seqlen))

m = Chain(
  LSTM(N, 128),
  LSTM(128, 128),
  Dense(128, N),
  softmax)

m = gpu(m)

function loss(xs, ys)
  l = sum(crossentropy.(m.(gpu.(xs)), gpu.(ys)))
  Flux.truncate!(m)
  return l
end

opt = ADAM(params(m), 0.01)
tx, ty = (gpu.(Xs[5]), gpu.(Ys[5]))
evalcb = () -> @show loss(tx, ty)

Flux.train!(loss, zip(Xs, Ys), opt,
            cb = throttle(evalcb, 30))

# Sampling
m = cpu(m)

function sample(m, alphabet, len; temp = 1)
  Flux.reset!(m)
  buf = IOBuffer()
  c = rand(alphabet)
  for i = 1:len
    write(buf, c)
    c = wsample(alphabet, m(onehot(c, alphabet)).data)
  end
  return String(take!(buf))
end

sample(m, alphabet, 1000) |> println

# evalcb = function ()
#   @show loss(Xs[5], Ys[5])
#   println(sample(deepcopy(m), alphabet, 100))
# end
