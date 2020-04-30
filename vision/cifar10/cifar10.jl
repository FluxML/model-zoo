using Flux, Metalhead, Statistics
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, flatten
using Metalhead: trainimgs
using Parameters: @with_kw
using Images: channelview
using Statistics: mean
using Base.Iterators: partition
using CUDAapi
if has_cuda()
    @info "CUDA is on"
    import CuArrays
    CuArrays.allowscalar(false)
end

@with_kw mutable struct Args
    batchsize::Int = 128
    throttle::Int = 10
    lr::Float64 = 3e-4
    epochs::Int = 50
    splitr_::Float64 = 0.1
end

# Function to convert the RGB image to Float64 Arrays
function getarray(X)
    Float32.(permutedims(channelview(X), (2, 3, 1)))
end

function get_processed_data(args)
    # Fetching the train and validation data and getting them into proper shape	
    X = trainimgs(CIFAR10)
    imgs = [getarray(X[i].img) for i in 1:40000]
    #onehot encode labels of batch
   
    labels = onehotbatch([X[i].ground_truth.class for i in 1:40000],1:10)
	
    train_pop = Int((1-args.splitr_)* 40000)
    train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:train_pop, args.batchsize)])
    valset = collect(train_pop+1:40000)
    valX = cat(imgs[valset]..., dims = 4) |> gpu
    valY = labels[:, valset] |> gpu
	
    val = (valX,valY)	
    return train, val
end

function get_test_data()
    # Fetch the test data from Metalhead and get it into proper shape.
    test = valimgs(CIFAR10)

    # CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs
    testimgs = [getarray(test[i].img) for i in 1:1000]
    testY = onehotbatch([test[i].ground_truth.class for i in 1:1000], 1:10) |> gpu
    testX = cat(testimgs..., dims = 4) |> gpu

    test = (testX,testY)
    return test
end

# VGG16 and VGG19 models
function vgg16()
    return Chain(
            Conv((3, 3), 3 => 64, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(64),
            Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(64),
            MaxPool((2,2)),
            Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(128),
            Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(128),
            MaxPool((2,2)),
            Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            MaxPool((2,2)),
            Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            MaxPool((2,2)),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            MaxPool((2,2)),
            flatten,
            Dense(512, 4096, relu),
            Dropout(0.5),
            Dense(4096, 4096, relu),
            Dropout(0.5),
            Dense(4096, 10)) |> gpu
end

function vgg19()
    return Chain(
            Conv((3, 3), 3 => 64, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(64),
            Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(64),
            MaxPool((2,2)),
            Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(128),
            Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(128),
            MaxPool((2,2)),
            Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            MaxPool((2,2)),
            Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            MaxPool((2,2)),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            MaxPool((2,2)),
            flatten,
            Dense(512, 4096, relu),
            Dropout(0.5),
            Dense(4096, 4096, relu),
            Dropout(0.5),
            Dense(4096, 10)) |> gpu
end

accuracy(x, y, m) = mean(onecold(cpu(m(x)), 1:10) .== onecold(cpu(y), 1:10))

function train(; kws...)
    # Initialize the hyperparameters
    args = Args(; kws...)
	
    # Load the train, validation data 
    train,val = get_processed_data(args)

    @info("Constructing Model")	
    # Defining the loss and accuracy functions
    m = vgg16()

    loss(x, y) = logitcrossentropy(m(x), y)

    ## Training
    # Defining the callback and the optimizer
    evalcb = throttle(() -> @show(loss(val...)), args.throttle)
    opt = ADAM(args.lr)
    @info("Training....")
    # Starting to train models
    Flux.@epochs args.epochs Flux.train!(loss, params(m), train, opt, cb = evalcb)

    return m
end

function test(m)
    test_data = get_test_data()

    # Print the final accuracy
    @show(accuracy(test_data..., m))
end

cd(@__DIR__)
m = train()
test(m)
