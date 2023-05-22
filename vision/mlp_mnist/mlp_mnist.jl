# Simple multi-layer perceptron, for the MNIST hand-written digits.
# This example does not use a GPU, it's small enough not to need one.

using Flux, MLDatasets, Statistics

# Our model is very simple: Its one "hidden layer" has 32 "neurons" each connected to every input pixel.
# Each has a sigmoid nonlinearity, and is connected to every "neuron" in the output layer.
# Finally, softmax produces probabilities, i.e. positive numbers which add up to 1:

model = Chain(Dense(28^2 => 32, sigmoid), Dense(32 => 10), softmax)

p1 = model(rand(Float32, 28^2))  # run model on random data shaped like an image

@show sum(p1) ≈1;

p3 = model(rand(Float32, 28^2, 3))  # ...or on a batch of 3 fake, random "images"

@show sum(p3; dims=1);  # all approx 1. Last dim is batch dim.

#===== DATA =====#

# Calling MLDatasets.MNIST() will dowload the dataset if necessary,
# and return a struct containing it.
# It takes a few seconds to read from disk each time, so do this once:

train_data = MLDatasets.MNIST()  # i.e. split=:train
test_data = MLDatasets.MNIST(split=:test)

# train_data.features is a 28×28×60000 Array{Float32, 3} of the images.
# We need a 2D array for our model. Let's combine the reshape needed with
# other pre-processing, in a function:

function simple_loader(data::MNIST; batchsize::Int=64)
    x2dim = reshape(data.features, 28^2, :)
    yhot = Flux.onehotbatch(data.targets, 0:9)
    Flux.DataLoader((x2dim, yhot); batchsize, shuffle=true)
end

# train_data.targets is a 60000-element Vector{Int}, of labels from 0 to 9.
# Flux.onehotbatch([0,1,9], 0:9) makes a matrix of 0 and 1.

simple_loader(train_data)  # returns a DataLoader, with first element a tuple like this:

x1, y1 = first(simple_loader(train_data)); # (784×64 Matrix{Float32}, 10×64 OneHotMatrix)

model(x1)  # x1 is the right shape for our model

y1  # y1 is the same shape as the model output.

@show Flux.crossentropy(model(x1), y1);  # This will be our loss function

#===== ACCURACY =====#

# We're going to log accuracy and loss during training. There's no advantage to
# calculating these on minibatches, since MNIST is small enough to do it at once.

function simple_accuracy(model, data::MNIST=test_data)
    (x, y) = only(simple_loader(data; batchsize=length(data)))  # make one big batch
    y_hat = model(x)
    iscorrect = Flux.onecold(y_hat) .== Flux.onecold(y)  # BitVector
    acc = round(100 * mean(iscorrect); digits=2)
end

@show simple_accuracy(model);  # accuracy about 10%, on training data, before training!

#===== TRAINING =====#

# Make a dataloader using the desired batchsize:

train_loader = simple_loader(train_data, batchsize = 256)

# Initialise storage needed for the Adam optimiser, with our chosen learning rate:

opt_state = Flux.setup(Adam(3e-4), model);

# Then train for 30 epochs, printing out details as we go:

for epoch in 1:30
    loss = 0.0
    for (x, y) in train_loader
        # Compute the loss and the gradients:
        l, gs = Flux.withgradient(m -> Flux.crossentropy(m(x), y), model)
        # Update the model parameters (and the Adam momenta):
        Flux.update!(opt_state, model, gs[1])
        # Accumulate the mean loss, just for logging:
        loss += l / length(train_loader)
    end

    if mod(epoch, 2) == 1
        # Report on train and test, only every 2nd epoch:
        train_acc = simple_accuracy(model, train_data)
        test_acc = simple_accuracy(model, test_data)
        @info "After epoch = $epoch" loss train_acc test_acc
    end
end

# This should get to about 94% accuracy.
# To do better, try using Dense(784 => 64, relu) instead.

#===== INSPECTION =====#

using ImageCore, ImageInTerminal

xtest, ytest = only(simple_loader(test_data, batchsize=length(test_data)));

# There are many ways to look at images, you won't need ImageInTerminal if working in a notebook.
# ImageCore.Gray is a special type, which interprets numbers between 0.0 and 1.0 as shades:

reshape(xtest[:,33], 28, 28) .|> Gray |> transpose

@show Flux.onecold(ytest, 0:9)[33];  # true label, should match!

# Now we can compare the model's probabilities, for the same input.
# This should be highest at the same number:

p10 = (0:9) .=> model(xtest[:, 33]);
display(p10)
