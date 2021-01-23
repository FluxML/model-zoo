using Flux
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
using Parameters: @with_kw
using Statistics: mean
using CUDA
using MLDatasets: CIFAR10
using MLDataPattern: splitobs

if has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

@with_kw mutable struct Args
    batchsize::Int = 128
    throttle::Int = 10
    lr::Float64 = 3e-4
    epochs::Int = 50
    splitr_::Float64 = 0.1
end

function get_processed_data(args)
    x, y = CIFAR10.traindata()

    (train_x, train_y), (val_x, val_y) = splitobs((x, y), at=1-args.splitr_)

    train_x = float(train_x)
    train_y = onehotbatch(train_y, 0:9)
    val_x = float(val_x)
    val_y = onehotbatch(val_y, 0:9)
    
    return (train_x, train_y), (val_x, val_y)
end

function get_test_data()
    test_x, test_y = CIFAR10.testdata()
   
    test_x = float(test_x)
    test_y = onehotbatch(test_y, 0:9)
    
    return test_x, test_y
end

# VGG16 and VGG19 models
function vgg16()
    Chain(
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
        Dense(4096, 10)
    )
end

function vgg19()
    Chain(
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
        Dense(4096, 10)
    )
end

function train(; kws...)
    # Initialize the hyperparameters
    args = Args(; kws...)
	
    if CUDA.has_cuda()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end
    
    # Load the train, validation data 
    train, val = get_processed_data(args)
    
    train_data = Flux.Data.DataLoader(train, batchsize=args.batchsize)
    val_data = Flux.Data.DataLoader(val, batchsize=args.batchsize)

    @info("Constructing Model")	
    # Defining the loss and accuracy functions
    m = vgg16() |> device

    loss(x, y) = logitcrossentropy(m(x), y)

    ## Training
    # Defining the optimizer
    opt = ADAM(args.lr)
    ps = Flux.params(m)

    @info("Training....")
    # Starting to train models
    for epoch in 1:args.epochs
        @info "Epoch $epoch"

        for (x, y) in train_data
            x, y = x |> device, y |> device

            gs = Flux.gradient(ps) do 
                loss(x, y)
            end

            Flux.update!(opt, ps, gs)
        end

        validation_loss = 0f0
        for (x, y) in val_data
            x, y = x |> device, y |> device
            validation_loss += loss(x, y)
        end
        validation_loss /= length(val_data)
        @show validation_loss
    end

    return m
end

function test(m; kws...)
    args = Args(kws...)

    if CUDA.has_cuda()
        device = gpu
    else
        device = cpu
    end
    
    test_data = get_test_data()
    test_data = Flux.Data.DataLoader(test_data, batchsize=64)

    correct, total = 0, 0
    for (x, y) in test_data
        x, y = x |> device, y |> device
        correct += sum(onecold(cpu(m(x)), 0:9) .== onecold(cpu(y), 0:9))
        total += size(y, 2)
    end
    test_accuracy = correct / total

    # Print the final accuracy
    @show test_accuracy
end

cd(@__DIR__)
m = train()
test(m)
