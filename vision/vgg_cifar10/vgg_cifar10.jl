using Flux
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
using Statistics: mean
using CUDA
using MLDatasets: CIFAR10
using MLUtils: splitobs, DataLoader

if CUDA.has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

function get_processed_data(args)
    x, y = CIFAR10(:train)[:]

    (train_x, train_y), (val_x, val_y) = splitobs((x, y), at=1-args.valsplit)

    train_x = float(train_x)
    train_y = onehotbatch(train_y, 0:9)
    val_x = float(val_x)
    val_y = onehotbatch(val_y, 0:9)
    
    return (train_x, train_y), (val_x, val_y)
end

function get_test_data()
    test_x, test_y = CIFAR10(:test)[:]
   
    test_x = float(test_x)
    test_y = onehotbatch(test_y, 0:9)
    
    return test_x, test_y
end

# VGG16 and VGG19 models
function vgg16()
    Chain([
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
    ])
end

Base.@kwdef mutable struct Args
    batchsize::Int = 128
    lr::Float32 = 3f-4
    epochs::Int = 50
    valsplit::Float64 = 0.1
end

function train(; kws...)
    # Initialize the hyperparameters
    args = Args(; kws...)
	
    # Load the train, validation data 
    train_data, val_data = get_processed_data(args)
    
    train_loader = DataLoader(train_data, batchsize=args.batchsize, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=args.batchsize)

    @info("Constructing Model")	
    m = vgg16() |> gpu

    loss(m, x, y) = logitcrossentropy(m(x), y)

    ## Training
    # Defining the optimizer
    opt = Flux.setup(Adam(args.lr), m)

    @info("Training....")
    # Starting to train models
    for epoch in 1:args.epochs
        @info "Epoch $epoch"

        for (x, y) in train_loader
            x, y = x |> gpu, y |> gpu
            gs = Flux.gradient(m -> loss(m, x, y), m)
            Flux.update!(opt, m, gs[1])
        end

        validation_loss = 0f0
        for (x, y) in val_loader
            x, y = x |> gpu, y |> gpu
            validation_loss += loss(m, x, y)
        end
        validation_loss /= length(val_loader)
        @show validation_loss
    end

    return m
end

function test(m; kws...)
    args = Args(kws...)

    test_data = get_test_data()
    test_loader = DataLoader(test_data, batchsize=args.batchsize)

    correct, total = 0, 0
    for (x, y) in test_loader
        x, y = x |> gpu, y |> gpu
        correct += sum(onecold(cpu(m(x))) .== onecold(cpu(y)))
        total += size(y, 2)
    end
    test_accuracy = correct / total

    # Print the final accuracy
    @show test_accuracy
end

if abspath(PROGRAM_FILE) == @__FILE__
    m = train()
    test(m)
end

