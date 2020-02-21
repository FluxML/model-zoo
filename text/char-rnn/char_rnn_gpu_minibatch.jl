using Flux
using Flux: onehot, chunk, batchseq, throttle, crossentropy
using StatsBase: wsample
using Base.Iterators: partition
using CuArrays
using CUDAnative: device!
using Random

ϵ = 1.0f-32

epochs = 2
batch_size = 50
sequence = 50
gpu_device = 0

device!(gpu_device)
CuArrays.allowscalar(false)

input_file = joinpath(dirname(@__FILE__),"input.txt")

isfile(input_file) ||
    download("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
             input_file)

text = collect(String(read(input_file)))
alphabet = [unique(text)...,'_']
text = map(ch -> Float32.(onehot(ch,alphabet)),text)
stop = Float32.(onehot('_',alphabet))

N = length(alphabet)
seqlen = sequence
nbatch = batch_size

Xs = collect(partition(batchseq(chunk(text, nbatch), stop), seqlen))
txt = circshift(text,-1)
txt[end] = stop
Ys = collect(partition(batchseq(chunk(txt, nbatch), stop), seqlen))

model = Chain(
  LSTM(N, 128),
  LSTM(128, 256),
  LSTM(256, 128),
  Dense(128, N),
  softmax)
  m = model |>gpu

opt = ADAM(0.01)
tx, ty = (Xs[5]|>gpu, Ys[5]|>gpu)

function loss(xx, yy)
  out = 0.0f0
  for (idx, x) in enumerate(xx)
    out += crossentropy(m(x) .+ ϵ, yy[idx])
  end
  Flux.reset!(m)
  out
end

idxs = length(Xs)
for epoch_idx in 1:epochs
  for (idx,(xs,ys)) in enumerate(zip(Xs, Ys))
    Flux.train!(loss, params(m), [(xs|>gpu,ys|>gpu)], opt)
    lss = loss(tx,ty)
    if idx % 10 == 0
      @info "epoch# $(epoch_idx)/$(epochs)-$(idx)/$(idxs) loss = $(lss)"
    end
  end
end

# Sampling
function sample(m, alphabet, len)
  m = cpu(m)
  Flux.reset!(m)
  buf = IOBuffer()
  c = rand(alphabet)
  for i = 1:len
    write(buf, c)
    c = wsample(alphabet, m(onehot(c, alphabet)))
  end
  return String(take!(buf))
end
@info sample(m, alphabet, 1000)
