using Flux, MLDatasets
using Flux: onehotbatch, onecold, DataLoader, flatten, OptimiserChain
using BSON:@save,@load
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true" 

# wrong in Flux 0.13.9
Flux._old_to_new(rule::ClipNorm) = Flux.Optimisers.ClipNorm(rule.thresh) 

# Also, quick test of train(epochs=10, images=128) shows increasing loss, not sure why.

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

"""
By default gives the full dataset, keyword images gives (for testing purposes) 
only the 1:images elements of the train set.
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


function train(; epochs=100, images=:)

    #params: warning, the training can be long with these params
    train_loader, test_loader = get_data(128; images)
    η = 3f-4
    in_channel = 3
    patch_size = 2
    kernel_size = 7
    dim = 128
    dimPL = 2
    depth = 18
    use_cuda = true

    #logging the losses
    train_save = zeros(epochs, 2)
    test_save = zeros(epochs, 2)

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

    opt = OptimiserChain(
            WeightDecay(1f-3), 
            ClipNorm(1f0),
            Adam(η),
            )
    state = Flux.setup(opt, model)

    for epoch in 1:epochs
        for (x,y) in train_loader
            x,y = x|>device, y|>device
            grads = gradient(m->Flux.logitcrossentropy(m(x), y, agg=sum), model)
            Flux.update!(state, model, grads[1])
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