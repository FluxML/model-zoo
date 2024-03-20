using Flux, BSON

using CUDA
# This turns many mistakes which would make GPU execution very slow into errors:
CUDA.allowscalar(false)

using MLDatasets
# This will silence questions from MLDatasets about whether to download new data:
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true" 

function make_convmixer(in_channels::Int, kernel_size::Int, patch_size::Int, dim::Int, depth::Int, N_classes::Int=10)
    Chain(
        Conv((patch_size, patch_size), in_channels => dim, gelu; stride=patch_size),
        BatchNorm(dim),
        [
            Chain(
                SkipConnection(Chain(
                    Conv((kernel_size, kernel_size), dim=>dim, gelu; pad=SamePad(), 
groups=dim),
                    BatchNorm(dim)
                ), +),
                Conv((1,1), dim=>dim, gelu),
                BatchNorm(dim),
            ) 
            for i in 1:depth
        ]...,
        AdaptiveMeanPool((1,1)),
        Flux.flatten,
        Dense(dim => N_classes)
    )
end

"""
By default gives the full dataset, keyword images gives (for testing purposes) 
only the 1:images elements of the training set.
"""
function get_data(batchsize::Int; dataset = MLDatasets.CIFAR10, images = :)

    # Loading Dataset
    if images isa Integer
        xtrain, ytrain = dataset(:train)[1:images]
        xtest, ytest = dataset(:test)[1:(images÷10)]
    else
        xtrain, ytrain = dataset(:train)[images]
        xtest, ytest = dataset(:test)[images]
    end

    # Reshape data to comply to Julia's (width, height, channels, batch_size) convention in case there are only 1 channel (eg MNIST)
    if ndims(xtrain)==3
        w = size(xtrain)[1]
        xtrain = reshape(xtrain, (w,w,1,:))
        xtest = reshape(xtest, (w,w,1,:))
    end
    
    ytrain = Flux.onehotbatch(ytrain, 0:9)
    ytest = Flux.onehotbatch(ytest, 0:9)

    train_loader = Flux.DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true)
    test_loader = Flux.DataLoader((xtest, ytest), batchsize=batchsize)

    return train_loader, test_loader
end

"""
This function is used only for logging, with either train_loader or test_loader.
"""
function loss_and_accuracy(model, dataloader)
    n = 0      # count images seen
    l = 0.0f0  # total loss
    c = 0      # count correct classifications

    for (x1, y1) in dataloader
        x, y = gpu(x1), gpu(y1)  # assuming model is on gpu too!
        z = model(x)
        l += Flux.logitcrossentropy(z, y, agg=sum)
        c += sum(Flux.onecold(z) .== Flux.onecold(y))
        n += size(x)[end]
    end
    return l/n, c/n  # mean loss, and accuracy
end

"""
Main function: loads data, creates model, trains it, then saves trained model in "model.bson"
Keywords `train(epochs=2, images=99)` will use only the first 99 images of the training set
(to check that it runs).
"""
function train(; epochs=100, images=:)
    train_loader, test_loader = get_data(128; images)

    # hyper-parameters. Note that training can be long with these!
    in_channel = 3
    patch_size = 2
    kernel_size = 7
    dim = 128
    depth = 18

    model = make_convmixer(in_channel, kernel_size, patch_size, dim, depth, 10) |> gpu

    # optimiser and its state:
    opt = OptimiserChain(
            WeightDecay(1f-3),  # L2 regularisation
            ClipNorm(1f0),
            Adam(3f-4),  # learning rate
            )
    state = Flux.setup(opt, model)

    # arrays for logging:
    train_save = zeros(epochs, 2)
    test_save = zeros(epochs, 2)

    for epoch in 1:epochs
        for (x1, y1) in train_loader
            x, y = gpu(x1), gpu(y1)
            grads = gradient(m -> Flux.logitcrossentropy(m(x), y; agg=sum), model)
            Flux.update!(state, model, grads[1])
        end

        # logging
        train_loss, train_acc = loss_and_accuracy(model, train_loader, device)
        test_loss, test_acc = loss_and_accuracy(model, test_loader, device)
        train_save[epoch, :] = [train_loss, train_acc]
        test_save[epoch, :] = [test_loss, test_acc]

        if epoch%5==0
            @info "Epoch $epoch" train_loss test_acc
        end
    end

    # save the fully trained model
    BSON.@save "model.bson" cpu(model)
    BSON.@save "losses.bson" train_save test_save
    # it's generally more robust to save just the state, like this, but 
    # JLD2.jldsave("model.jld2"; state = Flux.state(model |> cpu), train_save, test_save)

    return model
end

if abspath(PROGRAM_FILE) == @__FILE__
    # This was run `julia convmixer.jl`, rather than interactively e.g. `include("convmixer.jl")`
    train()
end
