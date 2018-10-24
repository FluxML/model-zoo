# Flux: A 60 Minute Blitz
# =====================

# This is a quick intro to Flux based on [PyTorch's
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
