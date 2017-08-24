using Flux, Flux.Optimise, MNIST, Base.Iterators
using Flux: onehot, mse

x, y = traindata()
y = hcat(map(y -> onehot(y, 0:9), y)...)

m = Chain(
  Linear(28^2, 32, σ),
  Linear(32, 10),
  softmax)

loss(x, y) = mse(m(x), y)

train!(loss, repeated((x,y), 100), sgd(m, 10))

m(x[:,1])
