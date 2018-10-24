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
# of models.
