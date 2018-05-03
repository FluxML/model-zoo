using Flux, Metalhead
using Metalhead: trainimgs
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: partition

# Function to convert the RGB image to Float64 Arrays

function getarray(X)
    img = Array{Float64,3}(32, 32, 3)
    for i = 1:32, j = 1:32
        img[i, j, :] = Float64.(getfield.(X[i, j], 1:3))
    end
    img
end

# Fetching the train and validation data and getting them into proper shape

X = trainimgs(CIFAR10)
imgs = [getarray(X[i].img) for i in 1:50000]
labels = onehotbatch([X[i].ground_truth.class for i in 1:50000],1:10)
train = gpu.([(cat(4, imgs[i]...), labels[:,i]) for i in partition(1:49000, 1000)])
valset = collect(49001:50000)
valX = cat(4, imgs[valset]...) |> gpu
valY = labels[:, valset] |> gpu

# Building the model

m = Chain(
    Conv((3, 3), 3 => 64, relu, pad=(1, 1), stride=(1, 1)),
    Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
    x -> maxpool(x, (2, 2)),
    Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
    Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
    x -> maxpool(x, (2, 2)),
    Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
    Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
    Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
    Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
    x -> maxpool(x, (2, 2)),
    Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
    Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
    Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
    Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
    x -> maxpool(x, (2, 2)),
    Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
    Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
    Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
    Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
    x -> maxpool(x, (2, 2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(512, 4096, relu),
    Dropout(0.5),
    Dense(4096, 4096, relu),
    Dropout(0.5),
    Dense(4096, 10),
    softmax) |> gpu

# Defining the loss and accuracy functions

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(argmax(m(x), 1:10) .== argmax(y, 1:10))

# Defining the callback and the optimizer

evalcb = throttle(() -> @show(accuracy(valX, valY)), 10)

opt = ADAM(params(m))

# Starting to train models

Flux.train!(loss, train, opt, cb = evalcb)

# Fetch the test data from Metalhead and get it into proper shape.
# CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs

test = valimgs(CIFAR10)

testimgs = [getarray(test[i].img) for i in 1:10000]
testY = onehotbatch([test[i].ground_truth.class for i in 1:10000], 1:10) |> gpu
testX = cat(4, testimgs...) |> gpu

# Print the final accuracy

@show(accuracy(testX, testY))
