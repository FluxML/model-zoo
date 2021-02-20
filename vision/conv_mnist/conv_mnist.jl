## Classification of MNIST dataset 
## with the convolutional neural network know as LeNet5.
## This script also combines various
## packages from the Julia ecosystem  with Flux.
using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Statistics, Random
using Logging: with_logger, global_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
import ProgressMeter
import MLDatasets
import DrWatson: savename, struct2dict
import BSON
using CUDA

# LeNet5 "constructor". 
# The model can be adapted to any image size
# and number of output classes.
function LeNet5(; imgsize=(28,28,1), nclasses=10) 
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
    return Chain(
            Conv((5, 5), imgsize[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            flatten,
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, nclasses)
          )
end

function get_data(args)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32, dir=args.datapath)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32, dir=args.datapath)

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest),  batchsize=args.batchsize)
    
    return train_loader, test_loader
end

loss(ŷ, y) = logitcrossentropy(ŷ, y)

function eval_loss_accuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]        
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end

## utility functions

num_params(model) = sum(length, Flux.params(model)) 

round4(x) = round(x, digits=4)


# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 3e-4             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 128      # batch size
    epochs = 20          # number of epochs
    seed = 0             # set seed > 0 for reproducibility
    cuda = true          # if true use cuda (if available)
    infotime = 1 	     # report every `infotime` epochs
    checktime = 5        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = false      # log training with tensorboard
    savepath = nothing    # results path. If nothing, construct a default path from Args. If existing, may overwrite
    datapath = joinpath(homedir(), "Datasets", "MNIST") # data path: change to your data directory 
end

function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.cuda && CUDA.has_cuda()
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    ## DATA
    train_loader, test_loader = get_data(args)
    @info "Dataset MNIST: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

    ## MODEL AND OPTIMIZER
    model = LeNet5() |> device
    @info "LeNet5 model: $(num_params(model)) trainable params"    
    
    ps = Flux.params(model)  

    opt = ADAM(args.η) 
    if args.λ > 0 
        opt = Optimiser(opt, WeightDecay(args.λ))
    end
    
    ## LOGGING UTILITIES
    if args.savepath == nothing
        experiment_folder = savename("lenet", args, scientific=4,
                    accesses=[:batchsize, :η, :seed, :λ]) # construct path from these fields
        args.savepath = joinpath("runs", experiment_folder)
    end
    if args.tblogger 
        tblogger = TBLogger(args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
        @info "TensorBoard logging at \"$(args.savepath)\""
    end
    
    function report(epoch)
        train = eval_loss_accuracy(train_loader, model, device)
        test = eval_loss_accuracy(test_loader, model, device)        
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
        if args.tblogger
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" loss=train.loss  acc=train.acc
                @info "test"  loss=test.loss   acc=test.acc
            end
        end
    end
    
    ## TRAINING
    @info "Start Training"
    report(0)
    for epoch in 1:args.epochs
        p = ProgressMeter.Progress(length(train_loader))

        for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                ŷ = model(x)
                loss(ŷ, y)
            end
            Flux.Optimise.update!(opt, ps, gs)
            ProgressMeter.next!(p)   # comment out for no progress bar
        end
        
        epoch % args.infotime == 0 && report(epoch)
        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)
            modelpath = joinpath(args.savepath, "model.bson") 
            let model=cpu(model), args=struct2dict(args)
                BSON.@save modelpath model epoch args
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
end

## Execution as a script
if abspath(PROGRAM_FILE) == @__FILE__ 
    train()
end
