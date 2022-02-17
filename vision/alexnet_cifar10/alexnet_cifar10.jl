using MLDatasets
using Flux, Statistics
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using MLDatasets
using Images
using ProgressBars


function get_data(args)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # load train and test dataset
    x_train, y_train = CIFAR10.traindata(Float32)
    x_test,  y_test  = CIFAR10.testdata(Float32)

    # reshape
    x_train = reshape(x_train, 32, 32, 3, :)
    x_test = reshape(x_test, 32, 32, 3, :)

    # resize
    x_train = imresize(x_train, (64, 64, 3))
    x_test = imresize(x_test, (64, 64, 3))

    # one-hot-encode the labels
    y_train = onehotbatch(y_train, 0:9)
    y_test = onehotbatch(y_test, 0:9)

    train_loader = DataLoader((x_train, y_train), batchsize=args.batchsize, partial=false, shuffle=true)
    test_loader = DataLoader((x_test, y_test), batchsize=args.batchsize, partial=false)

    return train_loader, test_loader

end


function loss_function(ŷ, y)
    logitcrossentropy(ŷ, y)
end


function evaluation_loss_accuracy(loader, model)

    loss, accuracy, counter = 0f0, 0f0, 0

    for (x,y) in loader
        ŷ = model(x)
        loss += loss_function(ŷ,y)
        accuracy += sum(onecold(ŷ) .== onecold(y))
        counter +=  1
    end

    return loss / counter, accuracy / counter
end

# AdaptiveMeanPool cannot be used with 64x64 because it will throw a 
# DimensionMismatch. You can use it with 256x256 or bigger images
# For more information refer to: https://github.com/FluxML/model-zoo/issues/334
function set_model(; imgsize=(64, 64,3), num_classes=10)
    return Chain(
                Conv((11, 11), imgsize[end]=>64, stride=(4,4), relu, pad=(2,2)),
                MaxPool((3, 3), stride=(2,2)),  
                Conv((5, 5), 64=>192, relu, pad=(2,2)),
                MaxPool((3, 3), stride=(2,2)),
                Conv((3, 3), 192=>384, relu, pad=(1,1)),
                Conv((3, 3), 384=>256, relu, pad=(1,1)),
                Conv((3, 3), 256=>256, relu, pad=(1,1)),
                MaxPool((3, 3), stride=(2,2)),
                # AdaptiveMeanPool((6, 6)), 
                flatten,
                Dropout(0.5),
                Dense(256*1*1, 4096, relu), # With AdaptiveMeanPool((6, 6)) set 256 * 6 * 6
                Dropout(0.5),
                Dense(4096, 4096, relu),
                Dense(4096, num_classes))
end


Base.@kwdef mutable struct Args
    η = 1e-4            # learning rate
    batchsize = 128     # batch size
    epochs = 10         # number of epochs
end


function train(; kws...)
    args = Args(; kws...) # collect options in a struct for convenience

    @info "Getting data..."
    
    train_loader, test_loader = get_data(args)

    model = set_model() # my implmentation
    # model = alexnet() # Metalhead's implmentation
    # model = LeNet5() # test pipeline


    ps = Flux.params(model)  
    opt = ADAM(args.η) 

    @info "Start training..."

    for epoch in 1:args.epochs

        for (x,y) in ProgressBar(train_loader)
            @assert size(x) == (64, 64, 3, args.batchsize) 
            @assert size(y) == (10, args.batchsize) 

            gs = gradient(() -> loss_function(model(x), y), ps) 
            Flux.Optimise.update!(opt, ps, gs)
        end

        train = evaluation_loss_accuracy(train_loader, model)
        test = evaluation_loss_accuracy(test_loader, model)

        println("Epoch $(epoch-1)")
        println("\t Train => loss = $(train[1]) \t acc = $(train[2])")
        println("\t Test => loss = $(test[1]) \t acc = $(test[2])")
    
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    train()
end