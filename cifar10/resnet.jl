# Importing the required functions from Flux

using Flux
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: partition

# Reading the training data and testing data from MLDatasets

using MLDatasets
train_X, train_Y = CIFAR10.traindata()
test_X, test_Y = CIFAR10.testdata()

validation_size = 100

# Getting the data in the proper format

train_x = convert(Array{Float64,4},float.(train_X))
labels = onehotbatch(train_Y, 0:9)
test_x = convert(Array{Float64,4},float.(test_X))
labels_test = onehotbatch(test_Y, 0:9)


# Partition the data into minibatches of 100 each

train = [(train_x[:,:,:,i], labels[:,i])
         for i in partition(1:50000, 100)]; 

println("Data obtained and converted to required format. Stating to build model")
# Building the model. This is a residual network with 3 residual blocks

m1 = Conv((3,3), 3=>64, relu, pad=(1,1), stride=(1,1))

m2 = Chain(
    x -> m1(x),
    Conv((3,3), 64=>64, relu, pad=(1,1), stride=(1,1)),
    Conv((3,3), 64=>64, relu, pad=(1,1), stride=(1,1)))

m3 = x -> m1(x) + m2(x)

m4 = Chain(
    x -> m3(x),
    Conv((3,3), 64=>64, relu, pad=(1,1), stride=(1,1)),
    Conv((3,3), 64=>64, relu, pad=(1,1), stride=(1,1)))

m5 = Chain(
    x -> m3(x) + m4(x),
    Conv((3,3), 64=>128, relu, pad=(1,1), stride=(1,1)))

m6 = Chain(
    x -> m5(x),
    Conv((3,3), 128=>128, relu, pad=(1,1), stride=(1,1)),
    Conv((3,3), 128=>128, relu, pad=(1,1), stride=(1,1)))

m7 = x -> m5(x) + m6(x)

m8 = Chain(
    x -> m7(x),
    Conv((3,3), 128=>128, relu, pad=(1,1), stride=(1,1)),
    Conv((3,3), 128=>128, relu, pad=(1,1), stride=(1,1)))

m9 = Chain(
    x -> m8(x) + m7(x),
    x -> meanpool(x, (2,2)),
    x -> reshape(x, :, size(x,4)),
    Dense(16*16*128, 10),
    softmax)

# Check if the model is built correctly. An error with will be thrown if the model defination is incorrect
# m9(train[1][1])

# Define the loss function. Note the use of m9(...). This is because m9 is the last block in the model.
loss(x, y) = crossentropy(m9(x), y)

# Checking if the loss function has been defined correctly
# loss(train[1][1],train[1][2])

# Define the accuracy metric. Here also m9 is used for the same reason
accuracy(x, y) = mean(argmax(m9(x)) .== argmax(y))

# Define the callbacks and the optimizer
evalcb = throttle(() -> @show(accuracy(test_x[:,:,:,1:validation_size], labels_test[:,1:validation_size])), 1)
opt = ADAM(params(m9))

println("Model has been successfully built. Starting to train")
# Start training the model
Flux.train!(loss, train, opt, cb = evalcb)