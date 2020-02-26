# # Classification of MNIST digits with a convolutional network.
# This program writes out saved model to the file "mnist_conv.bson".
# This file demonstrates basic implimentation of ~
# - data structure
# - construction
# - pooling
# - training
# - saving
# - conditional early-exit
# - learning rate scheduling.
#
#
# **This model, while simple, should hit around 99% test accuracy after training for approximately 20 epochs.**
#
# For core concepts of ML and Convolution check out [towardsdatascience's guide](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) for a detailed explanation of everything
#
# ## Importing required libraries
# Can't do machine learning in flux whithout `Flux` :)
using Flux
# Import the mnist dataset
using Flux.Data.MNIST
# `Statistics` to calculate the mean which is required for finding the accuracy
using Statistics
# The uses of these will be explained later
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
# `Printf` For outputing the accuracy and other information and `BSON` forsaving the model
using Printf, BSON

# ## Load labels and images from Flux.Data.MNIST
@info("Loading data set")
train_labels = MNIST.labels()
train_imgs = MNIST.images()

# ## Reshaping and batching
# The `train_imgs` is **not** of the right shape. So, it can't be directly divided into smaller parts for batching.
# For this, we are creating another array of the correct shape which will be our batch, and we'll fill this array with the data from our `train_imgs`.
# The correct shape, which Flux expects the batch to be in, is **(IMG_DATA, IMG_DATA, COLOR_CHANNEL, SAMPLES)**.
#
# A good way to think about this is that every image is a *2D* rectangle. Which has a hight and width of 28.
# Flux requires us to add another dim which represents the color. So a ***BLACK AND WHITE*** image would be like a cuboid of thickness 1, hight and width of 28.
# An ***RGB*** image would be like cuboid thickness 3, made up of 3 rectangles stackted on top of each other . each rectangle representing red, blue and green channels.
#
# This type of thinking is good for beginners as it helps visualize things.
#
# Note that we're adding image to the last (samples) dim
# So, a batch can be visualized as multiple cubiods, each of which is made up of *3* rectangles (R - rectangle, G - rectangle,)
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)

        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end

# In the above, onehotencoding turns neumerical data into kind of truth tabels.
# example,`onehot(:b, [:a, :b, :c])` whill output
# ```julia
# 3-element Flux.OneHotVector:
# false
#  true
# false
# ```
# onehotbatch just one hot encodes all the data in a provided array
# and yields the encoding of every element in an output *OneHotMatrix*.


batch_size = 128
mb_idxs = partition(1:length(train_imgs), batch_size)
train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs]

# We are dividing an array of numbers of the size of the imgs into the sizes of batches and saving it in `mb_idxs`.
# This **array of arrays** of numbers will be used as indexs.

# ## Prepare test set as one giant minibatch:
test_imgs = MNIST.images(:test)
test_labels = MNIST.labels(:test)
test_set = make_minibatch(test_imgs, test_labels, 1:length(test_imgs))

# ## Define our model.
# We will use a simple convolutional architecture with
# three iterations of Conv -> ReLU -> MaxPool, followed by a final Dense
# layer that feeds into a softmax probability output.
#
@info("Constructing model...")
model = Chain(

    Conv((3, 3), 1=>16, pad=(1,1), relu),
    MaxPool((2,2)),

    Conv((3, 3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),


    Conv((3, 3), 32=>32, pad=(1,1), relu),
    MaxPool((2,2)),


    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10),


    softmax,
)


# A conv layer is defined as `Conv((3, 3), 1=>16, pad=(1,1), relu)`
#
# `(3,3)` is the size of the filter that will be convolving.
# `1=>16` follows the `input_size=>output_size` format.
# `relu` is the name of the activation function we're gonna be using
#
# First convolution, operating upon a 28x28 image
# Second convolution, operating upon a 14x14 image
# Third convolution, operating upon a 7x7 image
# Then reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
# which is where we get the 288 in the `Dense`.
#
# Finally, softmax to get nice probabilities


# ## Generic Pre-processing
# Load model and datasets onto GPU, if enabled
train_set = gpu.(train_set)
test_set = gpu.(test_set)
model = gpu(model)

# Make sure our model is nicely precompiled before starting our training loop
model(train_set[1][1])

# `loss()` calculates the crossentropy loss between our prediction `y_hat`
# (calculated from `model(x)`) and the ground truth `y`.  We augment the data
# a bit, adding gaussian random noise to our image to make it more robust and avoid overfitting.
#
function loss(x, y)

    x_aug = x .+ 0.1f0*gpu(randn(eltype(x), size(x)))

    y_hat = model(x_aug)
    return crossentropy(y_hat, y)
end
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
# Onecold does what is sounds like it might do, it turns one hotencoded vectors into the orignal data.
#
# There are two reasons for using onecold.
# One is that m(x) has its last layer as `softmax`. This means it returns a probability distribution.
# We can not equate PB distributions. So, we need to turn the probablity distribution into neumerical data.
# As it turns out, onecold works with probablity distributions too.
# This means it turns the PB into normal neumerical data which can be used with the `==` sign.


opt = ADAM(0.001)
# Train our model with the given training set using the ADAM optimizer and
# printing out performance against the test set as we go.

# ## Training
@info("Beginning training loop...")
best_acc = 0.0
last_improvement = 0
for epoch_idx in 1:100
    global best_acc, last_improvement
    # Train for a single epoch
    Flux.train!(loss, params(model), train_set, opt)

    # Calculate accuracy:
    acc = accuracy(test_set...)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))

    # If our accuracy is good enough, quit out.
    if acc >= 0.999
        @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if acc >= best_acc
        @info(" -> New best accuracy! Saving model out to mnist_conv.bson")
        BSON.@save joinpath(dirname(@__FILE__), "mnist_conv.bson") model epoch_idx acc
        best_acc = acc
        last_improvement = epoch_idx
    end

    # If we haven't seen improvement in 5 epochs, drop our learning rate:
    if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
        opt.eta /= 10.0
        @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

        # After dropping learning rate, give it a few epochs to improve
        last_improvement = epoch_idx
    end

    if epoch_idx - last_improvement >= 10
        @warn(" -> We're calling this converged.")
        break
    end
end
