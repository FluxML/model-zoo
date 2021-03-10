using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using CUDA
using MLDatasets

if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

@kwdef mutable struct Args
    η::Float64 = 3e-4       # learning rate
    batchsize::Int = 1024   # batch size
    epochs::Int = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
end

function getdata(args)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset	
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)
	
    # Reshape Data in order to flatten each image into a linear array
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Batching
    train_data = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_data = DataLoader((xtest, ytest), batchsize=args.batchsize)

    return train_data, test_data
end

function build_model(; imgsize=(28,28,1), nclasses=10)
    return Chain(
 	        Dense(prod(imgsize), 32, relu),
            Dense(32, nclasses))
end

function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += logitcrossentropy(model(x), y)
    end
    return l / length(dataloader)
end

function accuracy(data_loader, model)
    acc = 0
    num = 0
    for (x, y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))
        num +=  size(x, 2)
    end
    return acc / num
end

function train(; kws...)
    # Initializing Model parameters 
    args = Args(; kws...)
    device = has_cuda() && args.use_cuda ? gpu : cpu
    # Load Data
    train_data,test_data = getdata(args)


    # Construct model
    m = build_model() |> device
    train_data = train_data |> device 
    test_data = test_data |> device
    
    # Define loss function 
    loss(x,y) = logitcrossentropy(m(x), y)
    
    ## Training
    evalcb = () -> @show(loss_all(train_data, m))
    opt = ADAM(args.η)
    
    # train for args.epochs epochs
    @epochs args.epochs Flux.train!(loss, params(m), train_data, opt, cb = evalcb)

    # After training, show accuracy for train and test set
    @show accuracy(train_data, m)
    @show accuracy(test_data, m)
end

cd(@__DIR__)
train()
# train(η=0.01) # can change hyperparameters
