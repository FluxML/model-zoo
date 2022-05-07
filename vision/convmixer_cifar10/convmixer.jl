using Flux, MLDatasets
using Flux: onehotbatch, onecold, DataLoader, Optimiser
using BSON:@save,@load


function ConvMixer(in_channels, kernel_size, patch_size, dim, depth, N_classes)
    f = Chain(
            Conv((patch_size, patch_size), in_channels=>dim, gelu; stride=patch_size),
            BatchNorm(dim),
            [
                Chain(
                    SkipConnection(Chain(Conv((kernel_size,kernel_size), dim=>dim, gelu; pad=SamePad(), groups=dim), BatchNorm(dim)), +),
                    Chain(Conv((1,1), dim=>dim, gelu), BatchNorm(dim))
                ) 
                for i in 1:depth
            ]...,
            AdaptiveMeanPool((1,1)),
            flatten,
            Dense(dim,N_classes)
        )
    return f
end

function get_data(batchsize; dataset = MLDatasets.CIFAR10, idxs = nothing)
    """
    idxs=nothing gives the full dataset, otherwise (for testing purposes) only the 1:idxs elements of the train set are given.
    """
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true" 

    # Loading Dataset
    if idxs===nothing
        xtrain, ytrain = dataset(:train)[:]
        xtest, ytest = dataset(:test)[:]
	else
        xtrain, ytrain = dataset(:train)[1:idxs]
        xtest, ytest = dataset(:test)[1:Int(idxs/10)]
    end

    # Reshape Data to comply to Julia's (width, height, channels, batch_size) convention in case there are only 1 channel (eg MNIST)
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

function create_loss_function(dataloader, device)

    function loss(model)
        n = 0
        l = 0.0f0
        acc = 0.0f0

        for (x,y) in dataloader
            x,y = x |> device, y |> device
            z = model(x)        
            l += Flux.logitcrossentropy(z, y, agg=sum)
            acc += sum(onecold(z).==onecold(y))
            n += size(x)[end]
        end
        l / n, acc / n
    end

    return loss
   
end


function train(n_epochs=100)

    #params: warning, the training can be long with these params
    train_loader, test_loader = get_data(128)
    η = 3e-4
    in_channel = 3
    patch_size = 2
    kernel_size = 7
    dim = 128
    dimPL = 2
    depth = 18
    use_cuda = true

    #logging the losses
    train_save = zeros(n_epochs, 2)
    test_save = zeros(n_epochs, 2)

    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    train_loss_fn = create_loss_function(train_loader, device)
    test_loss_fn = create_loss_function(test_loader, device)

    model = ConvMixer(in_channel, kernel_size, patch_size, dim, depth, 10) |> device

    ps = params(model)
    opt = Optimiser(
            WeightDecay(1f-3), 
            ClipNorm(1.0),
            ADAM(η)
            )

    for epoch in 1:n_epochs
        for (x,y) in train_loader
            x,y = x|>device, y|>device
            gr = gradient(()->Flux.logitcrossentropy(model(x), y, agg=sum), ps)
            Flux.Optimise.update!(opt, ps, gr)
        end

        #logging
        train_loss, train_acc = train_loss_fn(model) |> cpu
        test_loss, test_acc = test_loss_fn(model) |> cpu
        train_save[epoch,:] = [train_loss, train_acc]
        test_save[epoch,:] = [test_loss, test_acc]

        if epoch%5==0
            @info "Epoch $epoch : Train loss = $train_loss || Validation accuracy = $test_acc."
        end

    end

    model = model |> cpu
    @save "model.bson" model 
    @save "losses.bson" train_save test_save
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end