using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
# using CuArrays

# Classify MNIST digits with a simple multi-layer-perceptron
function demo(
    imgs = MNIST.images(),
    m = Chain(Dense(28^2, 32, relu), Dense(32, 10), softmax) |> gpu
)
  imgs = MNIST.images()
  # Stack images into one large batch
  X = reduce(hcat, float.(reshape.(imgs, :))) |> gpu

  labels = MNIST.labels()
  # One-hot-encode the labels
  Y = onehotbatch(labels, 0:9) |> gpu

  dataset = repeated((X, Y), 200)
  evalcb = () -> @show(loss(X, Y))
  opt = ADAM()
  loss(x, y) = crossentropy(m(x), y)
  accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))
  Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 10))

  @show accuracy(X, Y)

  # Test set accuracy
  tX = hcat(float.(reshape.(MNIST.images(:test), :))...) |> gpu
  tY = onehotbatch(MNIST.labels(:test), 0:9) |> gpu

  @show accuracy(tX, tY)
end

demo()
