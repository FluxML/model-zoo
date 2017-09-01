using Flux.Tracker

# This replicates the housing data example from the Knet.jl readme. Although we
# could have reused more of Flux (see the mnist example), the library's
# abstractions are very lightweight and don't force you into any particular
# strategy.

cd(@__DIR__)

isfile("housing.data") ||
  download("https://raw.githubusercontent.com/MikeInnes/notebooks/master/housing.data",
           "housing.data")

data = readdlm("housing.data")'

# The last feature is our target -- the price of the house.

x = data[1:13,:]
y = data[14:14,:]

# Normalise the data
x = (x .- mean(x,2)) ./ std(x,2)

# The model

W = track(randn(1,13)/10)
b = track([0.])

# using CuArrays
# W, b, x, y = cu.((W, b, x, y))

predict(x) = W*x .+ b
meansquarederror(ŷ, y) = sum((ŷ .- y).^2)/size(y, 2)
loss(x, y) = meansquarederror(predict(x), y)

function update!(ps, η = .1)
  for w in ps
    w.x .-= w.Δ .* η
    w.Δ .= 0
  end
end

for i = 1:10
  back!(loss(x, y))
  update!((W, b))
  @show loss(x, y)
end

predict(x[:,1]) / y[1]

# It's also easy to replicate Knet's `grad` approach:

function grad(f, xs...)
  xs = track.(xs)
  back!(f(xs...))
  Tracker.grad.(xs)
end

grad((W, b) -> meansquarederror(W*x.+b, y), rand(1,13), rand(1))
