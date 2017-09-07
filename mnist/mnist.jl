using Flux, MNIST
using Flux: onehotbatch, onecold, mse, throttle
using Base.Iterators: repeated

x, y = traindata()
y = onehotbatch(y, 0:9)

m = Chain(
  Dense(28^2, 32, σ),
  Dense(32, 10),
  softmax)

loss(x, y) = mse(m(x), y)

Flux.train!(loss, repeated((x,y), 1000), SGD(params(m), 0.1),
            cb = throttle(() -> @show(loss(x, y)), 5))

# Check the prediction for the first digit
onecold(m(x[:,1]), 0:9) == onecold(y[:,1], 0:9)
