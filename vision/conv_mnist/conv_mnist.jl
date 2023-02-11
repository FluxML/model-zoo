# Classification of MNIST dataset using a convnet, a variant of the original LeNet

using MLDatasets, Flux, CUDA, BSON  # this will install everything if necc.

#===== DATA =====#

tmp = MLDatasets.MNIST()

# This will dowload the dataset if necessary, and return a struct containing it:
# tmp.features is a 28×28×60000 Array{Float32, 3} of the images. 
# Flux needs images to be 4D arrays, with the 3rd dim for channels -- here trivial, grayscale.

function get_data(; split=:train, batchsize=64) # allows also split=:test
    x, y = MLDatasets.MNIST(; split)[:]
    x4dim = reshape(x, 28,28,1,:)  # insert channel dim
    yhot = Flux.onehotbatch(y, 0:9)
    isinf(batchsize) && return [(x4dim, yhot)] # |> gpu
    Flux.DataLoader((x4dim, yhot); batchsize, shuffle=true) # |> gpu
end

get_data(split=:test)  # returns a DataLoader, with first element a tuple like this:

x1, y1 = first(get_data()); # (28×28×1×64 Array{Float32, 3}, 10×64 OneHotMatrix(::Vector{UInt32}))

#===== MODEL =====#

# A layer like Conv((5, 5), 1=>6) takes 5x5 patches of an image, and matches them to
# each of 6 different 5x5 filters, placed at every possible position.

Conv((5, 5), 1=>6).weights |> summary  # 5×5×1×6 Array{Float32, 4}

# LeNet has two convolutional layers, and our modern version has relu nonlinearities.
# After each such layer, there's a pooling step, which keeps 1 result in each 2x2 window:

conv_layers = Chain(
    Conv((5, 5), 1=>6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6=>16, relu),
    MaxPool((2, 2)),
)

# Now conv_layers[1] is just the first Conv layer, conv_layers[1:2] includes the pooling layer.
# These can accept any size of image; let's trace the sizes with the actual input:

#=

julia> x1 |> size
(28, 28, 1, 64)

julia> conv_layers[1](x1) |> size
(24, 24, 6, 64)

julia> conv_layers[1:2](x1) |> size
(12, 12, 6, 64)

julia> conv_layers[1:3](x1) |> size
(8, 8, 16, 64)

julia> conv_layers(x1) |> size
(4, 4, 16, 64)

julia> conv_layers(x1) |> Flux.flatten |> size
(256, 64)

=#

# Flux.flatten is just reshape, preserving the batch dimesion (64) while combining others (4*4*16).
# These layers are going to be followed by some Dense layers, which need to know what size to expect.
# (See Flux.outputsize for ways to automate this.)

dense_layers = Chain(
    Dense(256 => 120, relu),
    Dense(120 => 84, relu), 
    Dense(84 => 10),
)

# Now assemble the whole network, and try it out:

lenet = Chain(conv_layers, Flux.flatten, dense_layers) # |> gpu

y1hat = lenet(x1)

softmax(y1hat)

# Each column of softmax(y1hat) may be thought of as the network's probabilities
# that an input image is in each of 10 classes. To find its most likely answer, 
# we can look for the largest output in each column, without needing softmax first. 
# At the moment, these don't resemble the true values at all:

hcat(Flux.onecold(y1hat, 0:9), Flux.onecold(y1, 0:9))

#===== METRICS =====#

# We're going to log accuracy and loss during training. There's no advantage to
# calculating these on minibatches, since MNIST is small enough to do it at once.

using Statistics: mean  # standard library

function loss_and_accuracy(model; split=:train)
    (x,y) = first(get_data(; split, batchsize=Inf))
    ŷ = model(x)
    loss = Flux.logitcrossentropy(ŷ, y)  # did not include softmax in the model
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    (; loss, acc, split)  # make a NamedTuple
end

loss_and_accuracy(lenet, split=:test)  # accuracy about 10%

#===== TRAINING =====#

# Let's collect some hyper-parameters in a NamedTuple, just to write them in one place.
# Global variables are fine -- we won't access this from inside any fast loops.

TRAIN = (;
    eta = 3e-4,     # learning rate
    lambda = 1e-2,  # for weight decay
    batchsize = 128,
    epochs = 10,
)
LOG = []

train_loader = get_data(batchsize=TRAIN.batchsize)

# Initialise the storage needed for the optimiser:

opt_rule = OptimiserChain(WeightDecay(TRAIN.lambda), Adam(TRAIN.eta))
opt_state = Flux.setup(opt_rule, lenet);

for epoch in 1:TRAIN.epochs
    @time for (x,y) in train_loader
        grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), lenet)
        Flux.update!(opt_state, lenet, grads[1])
    end

    # Logging & saving, not every epoch
    if epoch % 2 == 0
        loss, acc, _ = loss_and_accuracy(lenet)
        test_loss, test_acc, _ = loss_and_accuracy(lenet, split=:test)
        @info "logging:" epoch acc test_acc
        nt = (; epoch, loss, acc, test_loss, test_acc)
        push!(LOG, nt)
    end
    if epoch % 5 == 0
        name = joinpath("runs", "lenet.bson")
        # BSON.@save name lenet epoch
    end
end

LOG

loss_and_accuracy(lenet, split=:test)  # already logged

#===== INSPECTION =====#

using ImageInTerminal, ImageCore

xtest, ytest = first(get_data(; split=:test, batchsize=Inf))

# Many ways to look at images.
# ImageCore.Gray is a special type, whick interprets numbers between 0.0 and 1.0 as gray...

xtest[:,:,1,5] .|> Gray |> transpose  # should display a 4

Flux.onecold(ytest, 0:9)[5]  # it's a 4

# Let's look for the image whose classification is least certain.
# First, in each column of probabilities, ask for the largest one.
# Then, over all images, ask for the lowest such probability, and its index.

ptest = softmax(lenet(xtest))
max_p = maximum(ptest; dims=1)
_, i = findmin(vec(max_p))

xtest[:,:,1,i] .|> Gray |> transpose

Flux.onecold(ytest, 0:9)[i]  # true classification
# Flux.onecold(ptest[:,:,:,i:i], 0:9)  # uncertain prediction
# Maybe broken?

iwrong = findall(Flux.onecold(lenet(xtest)) .!= Flux.onecold(ytest))

xtest[:,:,1,itest[1]]
