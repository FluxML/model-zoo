# # Classification of MNIST dataset using ConvNet

# In this tutorial, we build a convolutional neural network (ConvNet or CNN) known as [LeNet5](https://en.wikipedia.org/wiki/LeNet) 
# to classify [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digits.

# LeNet5 is one of the earliest CNNs. It was originally used for recognizing handwritten characters. At a high level LeNet (LeNet-5) consists of two parts: 

# * A convolutional encoder consisting of two convolutional layers.
# * A dense block consisting of three fully-connected layers.

# The basic units in each convolutional block are a convolutional layer, a sigmoid activation function, 
# and a subsequent average pooling operation. Each convolutional layer uses a 5×5 kernel and a sigmoid activation function. 
# These layers map spatially arranged inputs to a number of two-dimensional feature maps, typically increasing the number of channels. 
# The first convolutional layer has 6 output channels, while the second has 16. 
# Each 2×2 pooling operation (stride 2) reduces dimensionality by a factor of 4 via spatial downsampling. 
# The convolutional block emits an output with shape given by (width, height, number of channels,  batch size).

# ![LeNet-5](../conv_mnist/docs/LeNet-5.png)

# Source: https://d2l.ai/chapter_convolutional-neural-networks/lenet.html

# >**Note:** The original architecture of Lenet5 used the sigmoind activation function. However, this is a a modernized version since it uses the RELU activation function instead.

# If you need more information about how CNNs work and related technical concepts, check out the following resources:

# * [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) . This is LeNet5 original paper by Yann LeCunn and others.
# * [Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/).
# * [Neural Networks in Flux.jl with Huda Nassar (working with the MNIST dataset)](https://youtu.be/Oxi0Pfmskus).
# * [Dive into Deep Learning", 2020](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html).


# This example demonstrates Flux’s Convolution and pooling layers, the usage of TensorBoardLogger, 
# how to write out the saved model to the file `mnist_conv.bson`, 
# and also combines various packages from the Julia ecosystem with Flux. 


# To run this example, we need the following packages:

using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
using Statistics, Random
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using ProgressMeter: @showprogress
import MLDatasets
import BSON
using CUDA

# We set default values for the arguments for the function `train`:

Base.@kwdef mutable struct Args
    η = 3e-4             ## learning rate
    λ = 0                ## L2 regularizer param, implemented as weight decay
    batchsize = 128      ## batch size
    epochs = 10          ## number of epochs
    seed = 0             ## set seed > 0 for reproducibility
    use_cuda = true      ## if true use cuda (if available)
    infotime = 1 	     ## report every `infotime` epochs
    checktime = 5        ## Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = true      ## log training with tensorboard
    savepath = "runs/"   ## results path
end

# ## Data

# We create the function `get_data` to load the MNIST train and test data from [MLDatasets](https://github.com/JuliaML/MLDatasets.jl) and reshape them so that they are in the shape that Flux expects. 

function get_data(args)
    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtest, ytest = MLDatasets.MNIST(:test)[:]

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest),  batchsize=args.batchsize)
    
    return train_loader, test_loader
end

# The function `get_data` performs the following tasks:

# * **Loads MNIST dataset:** Loads the train and test set tensors. The shape of the train data is `28x28x60000` and the test data is `28x28x10000`. 
# * **Reshapes the train and test data:**  Notice that we reshape the data so that we can pass it as arguments for the input layer of the model.
# * **One-hot encodes the train and test labels:** Creates a batch of one-hot vectors so we can pass the labels of the data as arguments for the loss function. For this example, we use the [logitcrossentropy](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.logitcrossentropy) function and it expects data to be one-hot encoded. 
# * **Creates mini-batches of data:** Creates two DataLoader objects (train and test) that handle data mini-batches of size `128 ` (as defined above). We create these two objects so that we can pass the entire data set through the loss function at once when training our model. Also, it shuffles the data points during each iteration (`shuffle=true`).

# ## Model

# We create the LeNet5 "constructor". It uses Flux's built-in [Convolutional and pooling layers](https://fluxml.ai/Flux.jl/stable/models/layers/#Convolution-and-Pooling-Layers):


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

# ## Loss function

# We use the function [logitcrossentropy](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.logitcrossentropy) to compute the difference between 
# the predicted and actual values (loss).

loss(ŷ, y) = logitcrossentropy(ŷ, y)

# Also, we create the function `eval_loss_accuracy` to output the loss and the accuracy during training:

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

# ## Utility functions
# We need a couple of functions to obtain the total number of the model's parameters. Also, we create a function to round numbers to four digits.

num_params(model) = sum(length, Flux.params(model)) 
round4(x) = round(x, digits=4)

# ## Train the model

# Finally, we define the function `train` that calls the functions defined above to train the model.

function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.use_cuda && CUDA.functional()
    
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
    if args.λ > 0 ## add weight decay, equivalent to L2 regularization
        opt = Optimiser(WeightDecay(args.λ), opt)
    end
    
    ## LOGGING UTILITIES
    if args.tblogger 
        tblogger = TBLogger(args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) ## 0 auto increment since we manually set_step!
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
        @showprogress for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                    ŷ = model(x)
                    loss(ŷ, y)
                end

            Flux.Optimise.update!(opt, ps, gs)
        end
        
        ## Printing and logging
        epoch % args.infotime == 0 && report(epoch)
        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)
            modelpath = joinpath(args.savepath, "model.bson") 
            let model = cpu(model) ## return model to cpu before serialization
                BSON.@save modelpath model epoch
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
end

# The function `train` performs the following tasks:

# * Checks whether there is a GPU available and uses it for training the model. Otherwise, it uses the CPU.
# * Loads the MNIST data using the function `get_data`.
# * Creates the model and uses the [ADAM optimiser](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.ADAM) with weight decay.
# * Loads the [TensorBoardLogger.jl](https://github.com/JuliaLogging/TensorBoardLogger.jl) for logging data to Tensorboard.
# * Creates the function `report` for computing the loss and accuracy during the training loop. It outputs these values to the TensorBoardLogger.
# * Runs the training loop using [Flux’s training routine](https://fluxml.ai/Flux.jl/stable/training/training/#Training). For each epoch (step), it executes the following:
#   * Computes the model’s predictions.
#   * Computes the loss.
#   * Updates the model’s parameters.
#   * Saves the model `model.bson` every `checktime` epochs (defined as argument above.)

# ## Run the example 

# We call the  function `train`:

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
