using Flux, Flux.Data.MNIST
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated

# Classify MNIST digits with a simple multi-layer-perceptron

imgs = MNIST.images()
# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...)

labels = MNIST.labels()
# One-hot-encode the labels
Y = onehotbatch(labels, 0:9)

m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax)

# using CuArrays
# x, y = cu(x), cu(y)
# m = mapleaves(cu, m)

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))

dataset = repeated((X, Y))
evalcb = () -> @show(loss(X, Y))

include("./learn-strat.jl")

opt = strategy(
  MaxIter(200),
  FluxModel(m, ADAM, throttle(evalcb, 1)))
# opt = ADAM(params(m))

LearningStrategies.learn!(loss, opt, dataset)
# Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))

accuracy(X, Y)

# Test set accuracy
tX = hcat(float.(reshape.(MNIST.images(:test), :))...)
tY = onehotbatch(MNIST.labels(:test), 0:9)
# If CuArrays
# tX, tY = cu(tX), cu(tY)
accuracy(tX, tY)
