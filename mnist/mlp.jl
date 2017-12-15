using Flux, MNIST
using Flux: onehotbatch, argmax, mse, throttle
using Base.Iterators: repeated

x, y = traindata()
y = onehotbatch(y, 0:9)

m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax)

# using CuArrays
# x, y = cu(x), cu(y)
# m = mapleaves(cu, m)

loss(x, y) = mse(m(x), y)

accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))

dataset = repeated((x, y), 200)
evalcb = () -> @show(loss(x, y))
opt = SGD(params(m), 0.1)

Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 5))

accuracy(x, y)

# Test set accuracy
tx, ty = testdata()
ty = onehotbatch(ty, 0:9)
accuracy(tx, ty)
