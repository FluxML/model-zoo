using Flux
using Flux.Batches: Tree, children, isleaf

include("data.jl")

N = length(alphabet)

# The recursive net itself. We can use it to combine sub-phrases into a single
# vector representing the combination.
W = param(randn(N, 2N)/100)
combine(a, b) = tanh.(W * [a; b])

# For example
# combine(rand(N), rand(N))
# combine(combine(rand(N), rand(N)), rand(N))
