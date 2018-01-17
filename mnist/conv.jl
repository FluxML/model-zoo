using Flux, Flux.Data.MNIST
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated, partition

# Classify MNIST digits with a convolutional network

imgs = MNIST.images()

labels = onehotbatch(MNIST.labels(), 0:9)

# Partition into batches of size 1,000
train = [(cat(4, float.(imgs[i])...), labels[:,i])
         for i in partition(1:60_000, 1000)]

# Prepare test set (first 1,000 images)
tX = cat(4, float.(MNIST.images(:test)[1:1000])...)
tY = onehotbatch(MNIST.labels(:test)[1:1000], 0:9)

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

evalcb = throttle(() -> @show(accuracy(tX, tY)), 10)
opt = ADAM(params(m))

Flux.train!(loss, train, opt, cb = evalcb)
