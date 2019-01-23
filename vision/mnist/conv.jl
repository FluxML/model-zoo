using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition

# Classify MNIST digits with a convolutional network

imgs = MNIST.images()

labels = onehotbatch(MNIST.labels(), 0:9)

# Partition into batches of size 32
train = [(cat(float.(imgs[i])..., dims = 4), labels[:,i])
         for i in partition(1:60_000, 32)]

train = gpu.(train)

# Prepare test set (first 1,000 images)
tX = cat(float.(MNIST.images(:test)[1:1000])..., dims = 4) |> gpu
tY = onehotbatch(MNIST.labels(:test)[1:1000], 0:9) |> gpu

m = Chain(
    Conv((3, 3), 1=>32, relu),
    Conv((3, 3), 32=>32, relu),
    x -> maxpool(x, (2,2)),
    Conv((3, 3), 32=>16, relu),
    x -> maxpool(x, (2,2)),
    Conv((3, 3), 16=>10, relu),
    x -> reshape(x, :, size(x, 4)),
    Dense(90, 10), softmax) |> gpu

m(train[1][1])

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

evalcb = throttle(() -> @show(accuracy(tX, tY)), 10)
opt = ADAM()

Flux.train!(loss, params(m), train, opt, cb = evalcb)
