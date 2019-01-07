# Model Zoo: Convolution Neural Network
# =====================

# This notebook is a quick overview of how to implement a 
# [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) 
# using [Flux](https://github.com/FluxML/Flux.jl) to classify handwritten characters from the 
# [MNIST](http://yann.lecun.com/exdb/mnist/) dataset

# MNIST is a collection of 70,000 images of handwritten numbers (0-9) at a resolution of 28x28.
# Those images are split into 60,000 training images and 10,000 testing images.

# ![A sample from the MNIST dataset](mnistsample.png "Some MNIST Entries")<center>A sample from the MNIST dataset</center>

# In this demo we will be using the training part of the dataset to train a convolutional neural
# network to identify which hand-written number is being shown. The network will then be evaluated
# against the testing images to give a score for how well it is able to classify the images.

# This demo has four steps:
# 0. Dependencies
# 1. Preparing the training data
# 2. Defining our convolutional neural network
# 3. Training and evaluating the model

# 0) Dependencies
# =====================

# This demo uses the standard `Flux` package, but also pulls in `Flux.Data.MNIST` to be able to use
# the MNIST dataset in a convenient form. The dataset could be pulled in from disk manually, but this is a nice and simple
# way to be able to get access to the data quickly.

# We also pull in `onehotbatch` and `onecold` for encoding the label data of the MNIST dataset. Lastly
# from Flux we pull in `crossentropy` and `throttle`. The `crossentropy` function is used in determining the loss at each iteration
# and `throttle` is used to control the rate of output to improve the user experience.

# The `Statisitics` package is used here for access to the 'mean' function and the final import of `partition` from Base is to aid 
# in the preparation of our test data.

using Statistics
using Flux, Flux.Data.MNIST
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: partition

# Optional extra:
# Using CuArrays will allow for us to place training data and the model onto the GPU and allow for much
# faster training and evaluation. CuArrays currently only supports NVidia GPUs. 
using CuArrays

# 1) Preparing the data
# ======================

# To begin, we start by importing the MNIST dataset through the functions made available by `Flux.Data.MNIST`.
# The images are taken as-is, but for the labels we use [`one-hot encoding`](https://github.com/FluxML/Flux.jl/blob/master/docs/src/data/onehot.md).
# One-hot encoding is a common technique in Machine Learning to simplfy complex data into a classification index for each possible result.
# This means changing the label from a single value for each label, such as `2` into a vector where only the second index is set. In our case we have 10 possible
# labels (0,1,2, ... 8,9), so each label is replaced with a vector with 10 entries and the corresponding index for the value is set to `1` and all other values to
# zero. So, an label value of `2` becomes `[0 1 0 0 0 0 0 0 0]`. One-hot encoding allows our classifier to predict how much an image looks like each of the possible
# outputs rather than simply having to predict a single value as an output.

imgs = MNIST.images()
labels = onehotbatch(MNIST.labels(), 0:9)

# Next, we can partition the training data into batches of 1,000 entries.
# This will allow us to do training with a subset of the data.
train = [(cat(float.(imgs[i])..., dims = 4), labels[:,i])
         for i in partition(1:60_000, 1000)]

# When using Flux with CuArrays we want our training data to be placed on the GPU. To do that we have the `gpu(x)` function
# which takes the data that is passed to it and places it on the GPU. This can later be reversed with a call to the `cpu(x)` function.
# Calling this function without CuArrays will do nothing.
train = gpu.(train)

# Now that the training data has been created, the test data needs to be created in the same way.
# In this case we are preparing 1,000 images to evaluate the model. The test images and labels are created
# in the same format as the training data, which means also one-hot encoding the test labels.

# In this code we use the `|>` operator. This operator takes everything before it and wraps it in the function after it.
# So, `10 |> sqrt` becomes sqrt(10). In our code we are using the `gpu` function to place all the test data on the GPU 
# in a non-obtrusive way.

# If you have not included the `using CuArrays` call because you are not running on a GPU then the 
# calls to `|> gpu` will pass through without attempting to place the code on the GPU, so do not worry about it being there
# if you are running on the CPU.
tX = cat(float.(MNIST.images(:test)[1:1000])..., dims = 4) |> gpu
tY = onehotbatch(MNIST.labels(:test)[1:1000], 0:9) |> gpu


# 2) Defining the convolutional network
# =====================================

# In Flux multiple layers are connected through the `Chain` function
# which allows for the layers to be applied in sequence on an input.
# To create a network with this function we simply pass each layer as
# an input.

# A convolutional layer in Flux is constructed with the `Conv` function.
# The `Conv` function takes a minimum of two parameters. 

# Firstly, the size of the convolutional
# mask as a tuple. In this demo we are using `(2,2)` which represents a 2x2 grid of weights. The second
# input is the mapping of input to output channels.

# The standard convolutional network setup involves applying the convolution to the input, then applying
# an `activation` to normalise the output of that convolution (in this demo we are using ReLU) and finally 
# the highest value in each convolutional grid is selected and passed through with the `maxpooling` layer.

# These three steps are shown in the first two inputs to the chain function below (the activation is added as the last input to the `Conv` call)

# As with the data above, we append the call to `Chain` with `|> gpu` which will compile these functions to
# the gpu and allow them to be used with the gpu data we have already setup.

m = Chain(
  Conv((2,2), 1=>16, relu),
  x -> maxpool(x, (2,2)),
  Conv((2,2), 16=>8, relu),
  x -> maxpool(x, (2,2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(288, 10), softmax) |> gpu

# In our demo we have two convolutional layers with max pooling and then a dense connection of the results
# to apply the weights to our 10 `one-hot` output.

# The next bit of code applies our training data to the model
# as well as constructing a simple loss function using `crossentropy`
# and an accuracy function we can use to track the progress of the training
# which compares the generated output from the current model for the inputs
# compared to the correct answers.

m(train[1][1])
loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))
opt = ADAM(params(m))


# 3) Training and evaluating the model
# =====================================

# With the model and data in place we can begin training.
# To train one epoch of the model we call the Flux `train!` function.
# this function takes the loss function, training data, our optimisation function
# and then an optional call back function.

# In this demo we use the callback function to evaluate the current model and output
# the performance as it is being trained.
evalcb = throttle(() -> @show(accuracy(tX, tY)), 10)
Flux.train!(loss, train, opt, cb = evalcb)

