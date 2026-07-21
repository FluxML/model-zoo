using Flux, Metalhead
using Flux: onehotbatch, onecold, crossentropy, throttle
using Metalhead: trainimgs
using Images: channelview
using Statistics: mean
using Base.Iterators: partition

squeeze(x, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes) = Conv((1,1), inplanes => squeeze_planes, relu, pad = (0,0), stride = (1,1))(x)

expand1x1(x, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes) = Conv((1,1), squeeze_planes => expand1x1_planes, relu, pad = (0,0), stride = (1,1))(x)

expand3x3(x, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes) = Conv((3,3), squeeze_planes => expand3x3_planes, relu, pad = (1,1), stride = (1,1))(x)

function Fire(x, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes)
  x = squeeze(x, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes)
  return cat(expand1x1(x, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes), expand3x3(x, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes), dims = 3)
end

Squeezenet() = Chain(Conv((7,7), 3 => 96, relu, pad = (0, 0), stride = (2,2)), 
  x -> maxpool(x, (3, 3), pad = (0,0), stride = (2,2)),
  x -> Fire(x, 96, 16, 64, 64),
  x -> Fire(x, 128, 16, 64, 64),
  x -> Fire(x, 128, 32, 128, 128),
  x -> maxpool(x, (3, 3), pad = (0,0), stride = (2,2)),
  x -> Fire(x, 256, 32, 128, 128),
  x -> Fire(x, 256, 48, 192, 192),
  x -> Fire(x, 384, 48, 192, 192),
  x -> Fire(x, 384, 64, 256, 256),
  x -> maxpool(x, (3, 3), pad = (0,0), stride = (2,2)),
  x -> Fire(x, 512, 64, 256, 256),
  Dropout(0.5),
  Conv((1,1), 512 => 10, relu, pad = (0,0), stride = (1,1)),
  x -> maxpool(x, (size(x,1), size(x,2))),
  x -> reshape(x, :, size(x,4)),
  softmax
  ) |> gpu

# Function to convert the RGB image to Float64 Arrays
getarray(X) = Float64.(permutedims(channelview(X), (2, 3, 1)))

# Fetching the train and validation data and getting them into proper shape

X = trainimgs(CIFAR10)
imgs = [getarray(X[i].img) for i in 1:50000]
labels = onehotbatch([X[i].ground_truth.class for i in 1:50000],1:10)
train = gpu.([(cat(imgs[i]..., dims=4), labels[:,i]) for i in partition(1:49000, 1000)])
valset = collect(49001:50000)
valX = cat(imgs[valset]..., dims = 4) |> gpu
valY = labels[:, valset] |> gpu

# Defining the loss and accuracy functions

m = Squeezenet()

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(onecold(m(x), 1:10) .== onecold(y, 1:10))

# Defining the callback and the optimizer

evalcb = throttle(() -> @show(accuracy(valX, valY)), 10)

opt = ADAM()

# Starting to train models

Flux.train!(loss, params(m), train, opt, cb = evalcb)

# Fetch the test data from Metalhead and get it into proper shape.
# CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs

test = valimgs(CIFAR10)

testimgs = [getarray(test[i].img) for i in 1:10000]
testY = onehotbatch([test[i].ground_truth.class for i in 1:10000], 1:10) |> gpu
testX = cat(testimgs..., dims=4) |> gpu

# Print the final accuracy

@show(accuracy(testX, testY))
