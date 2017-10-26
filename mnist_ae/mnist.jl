using Base.Iterators: repeated

using Flux
using Flux: onehotbatch, argmax, mse, throttle, accuracy
using Flux.Tracker
using MNIST

info("Loading train data")
x, _ = traindata()
x′ = x ./ 255

info("Building model")
include("./model.jl")

# info("using CuArrays")
# using CuArrays
# x′ = cu(x′)
# m = mapleaves(cu, m)

loss(x) = mse(x, m(x))

dataset = repeated((x′,), 200)
evalcb = () -> @show(loss(x′))
opt = ADAM(params(m), .005)

info("Training")
Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 1))

filename = "model.jld"
info("Saving model to $filename")
include("./io.jl")
savemodel(filename, m)

info("Output Sample image")
tx, _ = testdata()
outputimgs("sample.png", tx[:, rand(1:size(tx, 2), 20)])
