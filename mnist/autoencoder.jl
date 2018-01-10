using Flux
using Flux: onehotbatch, argmax, mse, throttle, accuracy
using Base.Iterators: partition
using MNIST

# Encode MNIST images as compressed vectors that can later be decoded back into
# images.

x, _ = traindata()
x ./= 255

# Partition into batches of size 1000
data = [(x[:,i],) for i in partition(1:60_000, 1000)]

N = 32 # Size of the encoding

m = Chain(
  Dense(28^2, N, relu), # Encoder
  Dense(N, 28^2, relu)) # Decoder

loss(x) = mse(m(x), x)

evalcb = () -> @show loss(data[1][1])
opt = ADAM(params(m))

for i = 1:10
  info("Epoch $i")
  Flux.train!(loss, data, opt, cb = throttle(evalcb, 5))
end

using Images

img(x::Vector) = Gray.(reshape(clamp.(x, 0, 1), 28, 28))

function sample()
  # 20 random digits
  xs = [x[:, i] for i in rand(1:size(x, 2), 20)]
  # Before and after images
  before, after = img.(xs), img.(map(x -> m(x).data, xs))
  # Stack them all together
  hcat(vcat.(before, after)...)
end

cd(@__DIR__)

save("sample.png", sample())
