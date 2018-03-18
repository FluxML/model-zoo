using MLDatasets
using Flux
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated, partition

# Fetching the train and test data and getting them into proper shape

train_X, train_Y = CIFAR10.traindata()
train_x = convert(Array{Float64,4},float.(train_X))
labels = onehotbatch(train_Y, 0:9)

test_X, test_Y = CIFAR10.testdata()
test_x = convert(Array{Float64,4},float.(test_X))
labels_test = onehotbatch(test_Y, 0:9)

train = [(train_x[:,:,:,i], labels[:,i])
         for i in partition(1:50000, 10)]

# Defining the blocks of convolution layers

conv1_1 = Conv((3,3), 3=>64, relu, pad=(1,1), stride=(1,1))
conv1_2 = Conv((3,3), 64=>64, relu, pad=(1,1), stride=(1,1))

conv2_1 = Conv((3,3), 64=>128, relu, pad=(1,1), stride=(1,1))
conv2_2 = Conv((3,3), 128=>128, relu, pad=(1,1), stride=(1,1))

conv3_1 = Conv((3,3), 128=>256, relu, pad=(1,1), stride=(1,1))
conv3_2 = Conv((3,3), 256=>256, relu, pad=(1,1), stride=(1,1))
conv3_3 = Conv((3,3), 256=>256, relu, pad=(1,1), stride=(1,1))

conv4_1 = Conv((3,3), 256=>512, relu, pad=(1,1), stride=(1,1))
conv4_2 = Conv((3,3), 512=>512, relu, pad=(1,1), stride=(1,1))
conv4_3 = Conv((3,3), 512=>512, relu, pad=(1,1), stride=(1,1))

conv5_1 = Conv((3,3), 512=>512, relu, pad=(1,1), stride=(1,1))
conv5_2 = Conv((3,3), 512=>512, relu, pad=(1,1), stride=(1,1))
conv5_3 = Conv((3,3), 512=>512, relu, pad=(1,1), stride=(1,1))

# Building the model

m = Chain(
    conv1_1,
    conv1_2,
    x -> maxpool(x, (2,2)),
    conv2_1,
    conv2_2,
    x -> maxpool(x, (2,2)),
    conv3_1,
    conv3_2,
    conv3_3,
    x -> maxpool(x, (2,2)),
    conv4_1,
    conv4_2,
    conv4_3,
    x -> maxpool(x, (2,2)),
    conv5_1,
    conv5_2,
    conv5_3,
    x -> maxpool(x, (2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(512, 4096, relu),
    Dropout(0.5),
    Dense(4096, 4096, relu),
    Dropout(0.5),
    Dense(4096, 10),
    softmax)

# Defining the loss and accuracy functions

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))

evalcb = throttle(() -> @show(accuracy(test_x, labels_test)), 1)
opt = ADAM(params(m))

# Starting to train models

Flux.train!(loss, train, opt, cb = evalcb)
