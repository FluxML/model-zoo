using Flux, MNIST
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated, partition

x, y = traindata()
x = reshape(x, 28, 28, 1, :)
y = onehotbatch(y, 0:9)

# Partition into batches of size 1000
train = [(x[:,:,:,i], y[:,i]) for i in partition(1:60_000, 1000)]

# Test set
tx, ty = testdata()
tx = reshape(tx, 28, 28, 1, :)
ty = onehotbatch(ty, 0:9)

m = Chain(
  Conv2D((2,2), 1=>16, relu),
  x -> maxpool2d(x, 2),
  Conv2D((2,2), 16=>8, relu),
  x -> maxpool2d(x, 2),
  x -> reshape(x, :, size(x, 4)),
  Dense(288, 10), softmax)

m(train[1][1])

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))

evalcb = throttle(() -> @show(accuracy(tx, ty)), 10)
opt = ADAM(params(m))

Flux.train!(loss, train, opt, cb = evalcb)
