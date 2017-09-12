using Flux, MNIST
using Flux: onehotbatch, argmax, mse, throttle
using Base.Iterators: repeated

x, y = traindata()
y = onehotbatch(y, 0:9)

m = Chain(
  Dense(28^2, 32, σ),
  Dense(32, 10),
  softmax)

loss(x, y) = mse(m(x), y)

dataset = repeated((x, y), 500)
evalcb() = @show(loss(x, y))
opt = SGD(params(m), 1)

Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))

# Check the prediction for the first digit
argmax(m(x[:,1]), 0:9) == argmax(y[:,1], 0:9)
