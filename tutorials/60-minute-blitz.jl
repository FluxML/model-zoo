# Deep Learning with Flux: A 60 Minute Blitz
# =====================

# This is a quick intro to [Flux](https://github.com/FluxML/Flux.jl) loosely
# based on [PyTorch's
# tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).
# It introduces basic Julia programming, as well as Flux's automatic
# differentiation (AD), which we'll use to build machine learning models. We'll
# use this to build a very simple neural network.

# Arrays
# -------

# The starting point for all of our models is the `Array` (sometimes referred to
# as a `Tensor` in other frameworks). This is really just a list of numbers,
# which might be arranged into a shape like a square. Let's write down an array
# with three elements.

x = [1, 2, 3]

# Here's a matrix – a square array with four elements.

x = [1 2; 3 4]

# We often work with arrays of thousands of elements, and don't usually write
# them down by hand. Here's how we can create an array of 5×3 = 15 elements,
# each a random number from zero to one.

x = rand(5, 3)

# There's a few functions like this; try replacing `rand` with `ones`, `zeros`,
# or `randn` to see what they do.

# By default, Julia works stores numbers is a high-precision format called
# `Float64`. In ML we often don't need all those digits, and can ask Julia to
# work with `Float32` instead. We can even ask for more digits using `BigFloat`.

x = rand(BigFloat, 5, 3)
#-
x = rand(Float32, 5, 3)

# We can ask the array how many elements it has.

length(x)

# Or, more specifically, what size it has.

size(x)

# We sometimes want to see some elements of the array on their own.

x
#-
x[2, 3]

# This means get the second row and the third column. We can also get every row
# of the third column.

x[:, 3]

# We can add arrays, and subtract them, which adds or subtracts each element of
# the array.

x + x
#-
x - x

# Julia supports a feature called *broadcasting*, using the `.` syntax. This
# tiles small arrays (or single numbers) to fill bigger ones.

x .+ 1

# We can see Julia tile the column vector `1:5` across all rows of the larger
# array.

zeros(5,5) .+ (1:5)

# The x' syntax is used to transpose a column `1:5` into an equivalent row, and
# Julia will tile that across columns.

zeros(5,5) .+ (1:5)'

# We can use this to make a times table.

(1:5) .* (1:5)'

# Finally, and importantly for machine learning, we can conveniently do things like
# matrix multiply.

W = randn(5, 10)
x = rand(10)
W * x

# Julia's arrays are very powerful, and you can learn more about what they can
# do [here](https://docs.julialang.org/en/v1/manual/arrays/).

# ### CUDA Arrays

# CUDA functionality is provided separately by the [CuArrays
# package](https://github.com/JuliaGPU/CuArrays.jl). If you have a GPU and CUDA
# available, you can run `] add CuArrays` in a REPL or IJulia to get it.

# Once CuArrays is loaded you can move any array to the GPU with the `cu`
# function, and it supports all of the above operations with the same syntax.

## using CuArrays
## x = cu(rand(5, 3))

# Automatic Differentiation
# -------------------------

# You probably learned to take derivatives in school. We start with a simple
# mathematical function like

f(x) = 3x^2 + 2x + 1

f(5)

# In simple cases it's pretty easy to work out the gradient by hand – here it's
# `6x+2`. But it's much easier to make Flux do the work for us!

using Flux.Tracker: derivative

df(x) = derivative(f, x)

df(5)

# You can try this with a few different inputs to make sure it's really the same
# as `6x+2`. We can even do this multiple times (but the second derivative is a
# fairly boring `6`).

ddf(x) = derivative(df, x)

ddf(5)

# Flux's AD can handle any Julia code you throw at it, including loops,
# recursion and custom layers, so long as the mathematical functions you call
# are differentiable. For example, we can differentiate a Taylor approximation
# to the `sin` function.

mysin(x) = sum((-1)^k*x^(1+2k)/factorial(1+2k) for k in 0:5)

x = 0.5

mysin(x), derivative(mysin, x)
#-
sin(x), cos(x)

# You can see that the derivative we calculated is very close to `cos(x)`, as we
# expect.

# This gets more interesting when we consider functions that take *arrays* as
# inputs, rather than just a single number. For example, here's a function that
# takes a matrix and two vectors (the definition itself is arbitrary)

using Flux.Tracker: gradient

myloss(W, b, x) = sum(W * x .+ b)

W = randn(3, 5)
b = zeros(3)
x = rand(5)

gradient(myloss, W, b, x)

# Now we get gradients for each of the inputs `W`, `b` and `x`, which will come
# in handy when we want to train models.

# Because ML models can contain hundreds of parameters, Flux provides a slightly
# different way of writing `gradient`. We instead mark arrays with `param` to
# indicate that we want their derivatives.

using Flux.Tracker: param, back!, grad

W = param(randn(3, 5))
b = param(zeros(3))
x = rand(5)

y = sum(W * x .+ b)

# Anything marked `param` becomes *tracked*, indicating that Flux keeping an eye
# on its gradient. We can now call

back!(y) # Run backpropagation

grad(W), grad(b)

# We can now grab the gradients of `W` and `b` directly from those parameters.

# This comes in handy when working with *layers*. A layer is just a handy
# container for some parameters. For example, `Dense` does a linear transform
# for you.

using Flux

m = Dense(10, 5)

x = rand(10)

m(x)
#-
m(x) == m.W * x .+ m.b

# We can easily get the parameters of any layer or model with params with
# `params`.

params(m)

# This makes it very easy to do backpropagation and get the gradient for all
# parameters in a network, even if it has many parameters.

m = Chain(Dense(10, 5, relu), Dense(5, 2), softmax)

l = sum(Flux.crossentropy(m(x), [0.5, 0.5]))
back!(l)

grad.(params(m))

# You don't have to use layers, but they can be convient for many simple kinds
# of models and fast iteration.

# Neural Networks
# ----------------

# Let's check out how to build a simple neural network using `Flux`.

# Here's an overview of the network we will construct. 

# ![title](https://pytorch.org/tutorials/_images/mnist.png)

# It is a simple feed-forward network. It takes the input, feeds it through several
# layers one after the other, and then finally gives the output. We will use the
# MNIST dataset for this blitz.

using Images
using Flux: onehotbatch
using Flux.Data: MNIST
labels = onehotbatch(MNIST.labels(), 0:9)
images = MNIST.images()
images[1:3]

# A typical training procedure for a neural network is as follows:

# Define the neural network that has some learnable parameters (or weights)
# * Iterate over a dataset of inputs
# * Process input through the network
# * Compute the loss (how far is the output from being correct)
# * Propagate gradients back into the network’s parameters
# * Update the weights of the network, typically using a simple update rule:
# `weights = weights - learning_rate * gradient`

# Let's use the layers API to build the net.

model = Chain(
  Conv((5,5), 1=>6, relu),
  x -> maxpool(x, (2,2)),
  Conv((5,5), 6=>16, relu),
  x -> maxpool(x, (2,2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(256, 120), 
  Dense(120, 84),
  Dense(84, 10), 
  softmax) |> gpu

# That's about it. We don't have to use the layers API, or could define our
# own layers. Let us see what a forward pass would look like on this net.

model(rand(28, 28, 1, 16))

# We are feeding 16 random 28 x 28 matrices, single channel "images".
# Flux knows to play well with the Julia ecosystem. We are using normal julia
# arrays here, and could just as easily have passed in our images. 

train = cat(float.(images[1:10])..., dims = 4)
model(train)

# Before we start training, we need to set everything that goes around the net.

# LOSS Function
# -------------

# A loss function takes the (output, target) pair of inputs, and computes a value
# that estimates how far away the output is from the target.

# Flux comes with some predefined loss functions, but these are simple functions too.
# A mean squared loss (as defined in the Flux [source code]
# (https://github.com/FluxML/Flux.jl/blob/a32c8a2e60870d49475719a36c74afed305e370a/src/layers/stateless.jl#L5))
# looks exactly like we would write it mathematically. You could define the loss any
# way you want. The loss used here is the one we have defined. Feel free to play around
# with it.

mymse(ŷ, y) = sum((ŷ .- y).^2)/length(y)
loss(x, y) = mymse(model(x), y)
l = loss(train, labels[:, 1:10])
#-

# Backpropagation
# ---------------

# Now we can backpropagate the gradients, which is a simple affair of calling `back!` on
# the loss. 

Tracker.back!(l) 

# We have our gradients accumulated and are ready to update the weights now.
# As you might be already familiar with, *Gradient Descent* relies on a very simple
# update rule.

# ```julia
# weights = weights - learning_rate * gradient
# ```

# Updateing the weights
# ---------------------

# Modern ML models typically rely on a bit more complicated algorithms to carry out 
# optimisation. Be it Momentum, RMSPRop or the many variations of ADAM, Flux comes with
# some of the popular ones, just for convenience. Using them and even writing our own
# is really simple.

opt = Descent(0.01)

# The optimisers hold their parameters and state where applicable. This is a julia
# `struct`. We just have to tell Flux the parameters that need to be updated. This is
# the same `params` that we saw ealier.

Flux.Optimise: update!
ps = params(model)
update!(opt, ps)

# Large ML applications generally require updating the parameters of the optimisers
# midway during the training. Since our optmisers are just structs, we can change the
# parameters at will

opt.eta = 0.02

# You could combine many optimisers, schedule learning rates, apply decay and so on
# as well with what Flux refers to as `Optimiser`.

o = Opitmiser(ADAM(0.01), ExpDecay(), Descent(0.1))
update!(o[1:2], ps) # to use only some part of the opitmiser

# Training
# --------

# Combining all these steps put together, done man times is how we train Flux models.
# Flux comes with the `train!` to start our training loop.

Flux.train!(loss, params(model), train, opt)

# You don't have to use `train!`. In cases where aribtrary logic might be better suited,
# you could open up this training loop like so:

for d in data # assuming d looks like (batch, labels)
	# our super logic
	l = loss(d...)
	Tracker.back!(l)
	update!(opt, params(model))
end

# Training a Classifier
# ---------------------

