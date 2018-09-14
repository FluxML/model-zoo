using Flux.Tracker, Statistics, DelimitedFiles
using Flux.Tracker: Params, gradient, update!

# This replicates the housing data example from the Knet.jl readme. Although we
# could have reused more of Flux (see the mnist example), the library's
# abstractions are very lightweight and don't force you into any particular
# strategy.

cd(@__DIR__)

isfile("housing.data") ||
  download("https://raw.githubusercontent.com/MikeInnes/notebooks/master/housing.data",
           "housing.data")

rawdata = readdlm("housing.data")'

# The last feature is our target -- the price of the house.

x = rawdata[1:13,:]
y = rawdata[14:14,:]

# Normalise the data
x = (x .- mean(x, dims = 2)) ./ std(x, dims = 2)

# The model

W = param(randn(1,13)/10)
b = param([0.])

# using CuArrays
# W, b, x, y = cu.((W, b, x, y))

predict(x) = W*x .+ b
meansquarederror(ŷ, y) = sum((ŷ .- y).^2)/size(y, 2)
loss(x, y) = meansquarederror(predict(x), y)

η = 0.1
θ = Params([W, b])

for i = 1:10
  g = gradient(() -> loss(x, y), θ)
  for x in θ
    update!(x, g[x]*η)
  end
  @show loss(x, y)
end

predict(x[:,1]) / y[1]
