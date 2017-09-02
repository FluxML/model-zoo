using Flux, Flux.Optimise, MNIST, Base.Iterators
using Flux: onehot, onecold, mse, throttle

x, y = traindata()
y = hcat(map(y -> onehot(y, 0:9), y)...)

m = Chain(
  Linear(28^2, 32, σ),
  Linear(32, 10),
  softmax)

loss(x, y) = mse(m(x), y)

train!(loss, repeated((x,y), 1000), SGD(m, 0.1),
       cb = throttle(() -> @show(loss(x, y)), 5))

# Check the prediction for the first digit
onecold(m(x[:,1]), 0:9) == onecold(y[:,1], 0:9)
