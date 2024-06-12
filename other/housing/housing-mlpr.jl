using Flux, Statistics, DelimitedFiles
using Flux: mse, throttle, gpu, @epochs
using Base.Iterators: repeated

cd(@__DIR__)

isfile("housing.data") || download("https://raw.githubusercontent.com/MikeInnes/notebooks/master/housing.data", "housing.data")

rawdata = readdlm("housing.data")'

x = rawdata[1:13,:] |> gpu
y = rawdata[14:14,:] |> gpu

# Normalise the data
x = (x .- mean(x, dims = 2)) ./ std(x, dims = 2)

# The model
m = Chain(
  Dense(13, 32, relu),
  Dropout(0.1),
  Dense(32, 1, relu), identity) |> gpu

loss(x, y) = mse(m(x), y)

dataset = repeated((x, y), 1000)
evalcb = () -> @show loss(x, y)
opt = ADAM()

@epochs 10 Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb,10))
