using Flux, CUDA, MLDatasets
using Flux: onehotbatch, onecold, logitcrossentropy, DataLoader, flatten, OptimiserChain
using BSON:@save,@load

# This will silence questions from MLDatasets about whether to download new data:
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true" 

function ConvMixer(in_channels, kernel_size, patch_size, dim, depth, N_classes)
    Chain(
        Conv((patch_size, patch_size), in_channels => dim, gelu; stride=patch_size),
        BatchNorm(dim),
        [
            Chain(
                SkipConnection(Chain(
                    Conv((kernel_size,kernel_size), dim=>dim, gelu; pad=SamePad(), 
groups=dim),
                    BatchNorm(dim)
                ), +),
                Conv((1,1), dim=>dim, gelu),
                BatchNorm(dim),
            ) 
            for i in 1:depth
        ]...,
        AdaptiveMeanPool((1,1)),
        flatten,
        Dense(dim => N_classes)
    )
end

"""
By default gives the full dataset, keyword images gives (for testing purposes) 
only the 1:images elements of the training set.
"""
function get_data(batchsize; dataset = MLDatasets.CIFAR10, images = :)

    # Loading Dataset
    if images === (:)
        xtrain, ytrain = dataset(:train)[:]
        xtest, ytest = dataset(:test)[:]
    else
        xtrain, ytrain = dataset(:train)[1:images]
        xtest, ytest = dataset(:test)[1:(images÷10)]
    end

    # Reshape data to comply to Julia's (width, height, channels, batch_size) convention in 
case there are only 1 channel (eg MNIST)
    if ndims(xtrain)==3
        w = size(xtrain)[1]
        xtrain = reshape(xtrain, (w,w,1,:))
        xtest = reshape(xtest, (w,w,1,:))
    end
    
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=batchsize)

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
        x, y = gpu(x1), gpu(y1)
        z = model(x)
        l += logitcrossentropy(z, y, agg=sum)
        c += sum(onecold(z).==onecold(y))
        n += size(x)[end]
    end
    return l/n, c/n  # mean loss, and accuracy
end

"""
Main function: loads data, creates model, trains it, then saves trained model in "model.bson"
Keyword `images=99` will use only the first 99 images of the training set (to check that it 
runs).
"""
function train(; epochs=100, images=:)
    train_loader, test_loader = get_data(128; images)

    # hyper-parameters. Note that training can be long with these!
    in_channel = 3
    patch_size = 2
    kernel_size = 7
    dim = 128
    depth = 18
    η = 3f-4  # learning rate

    model = ConvMixer(in_channel, kernel_size, patch_size, dim, depth, 10) |> gpu

    opt = OptimiserChain(
            WeightDecay(1f-3), 
            ClipNorm(1f0),
            Adam(η),
            )
    state = Flux.setup(opt, model)

    # arrays for logging:
    train_save = zeros(epochs, 2)
    test_save = zeros(epochs, 2)

    for epoch in 1:epochs
        for (x1, y1) in train_loader
            x, y = gpu(x1), gpu(y1)
            grads = gradient(m -> logitcrossentropy(m(x), y, agg=sum), model)
            Flux.update!(state, model, grads[1])
        end

        # logging
        train_loss, train_acc = loss_and_accuracy(model, train_loader, device)
        test_loss, test_acc = loss_and_accuracy(model, test_loader, device)
        train_save[epoch,:] = [train_loss, train_acc]
        test_save[epoch,:] = [test_loss, test_acc]

        if epoch%5==0
            @info "Epoch $epoch" train_loss test_acc
        end
    end

    @save "model.bson" cpu(model)
    @save "losses.bson" train_save test_save
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
