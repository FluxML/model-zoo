using Flux, MNIST
using Flux: onehotbatch, argmax, mse, throttle
using Base.Iterators: repeated

info("Loading train data")
x, y = traindata()
y = onehotbatch(y, 0:9)

info("Building model")
m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax)
@show m

# using CuArrays
# x, y = cu(x), cu(y)
# m = mapleaves(cu, m)

loss(x, y) = mse(m(x), y)

dataset = repeated((x, y), 200)
evalcb = () -> @show(loss(x, y), accuracy(x, y))
accuracy(x, y) = mean(argmax(m(x), 0:9) .== argmax(y, 0:9))
opt = SGD(params(m), 0.1)

info("Training")
Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 5))

info("Training finished")
@show accuracy(x, y)

# Check the prediction for the first digit
@assert argmax(m(x[:, 1]), 0:9)[] == argmax(y[:, 1], 0:9)

info("Testing accuracy")
tx, ty = testdata()
@show accuracy(tx, onehotbatch(ty, 0:9))
