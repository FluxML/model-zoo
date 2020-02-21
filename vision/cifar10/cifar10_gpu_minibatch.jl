# Julia version : 1.3.1
# Flux version : v0.10.1

using Random
using Dates
using CuArrays
using CUDAdrv
using CUDAnative: device!
using Flux, Metalhead, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Metalhead: trainimgs
using Images: channelview
using Statistics: mean
using Base.Iterators: partition

model_file = joinpath(dirname(@__FILE__),"cifar10_vgg16_model.bson")

# Get arguments

epochs = 100
batch_size = 128
gpu_device = 0

# Very important : this prevent loss NaN
ϵ = 1.0f-32

# use 1nd GPU
#CUDAnative.device!(0)
device!(gpu_device)
CuArrays.allowscalar(false)

# VGG16 and VGG19 models
vgg16() = Chain(
  Conv((3, 3), 3 => 64, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(64),
  Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(64),
  MaxPool((2,2)),
  Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(128),
  Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(128),
  MaxPool((2,2)),
  Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256),
  Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256),
  Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256),
  MaxPool((2,2)),
  Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  MaxPool((2,2)),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  MaxPool((2,2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(512, 4096, relu),
  Dropout(0.5),
  Dense(4096, 4096, relu),
  Dropout(0.5),
  Dense(4096, 10),
  softmax)

vgg19() = Chain(
  Conv((3, 3), 3 => 64, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(64),
  Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(64),
  MaxPool((2,2)),
  Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(128),
  Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(128),
  MaxPool((2,2)),
  Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256),
  Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256),
  Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256),
  Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
  MaxPool((2,2)),
  Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  MaxPool((2,2)),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  MaxPool((2,2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(512, 4096, relu),
  Dropout(0.5),
  Dense(4096, 4096, relu),
  Dropout(0.5),
  Dense(4096, 10),
  softmax)

m = vgg16() |> gpu

# Function to convert the RGB image to Float32 Arrays
getarray(X) = Float32.(permutedims(channelview(X), (2, 3, 1)))

function make_minibatch(imgs,labels,batch_size)
  data_set = [(cat(imgs[i]..., dims = 4),
          labels[:,i])
          for i in partition(1:length(imgs), batch_size)]
  return data_set
end

X = trainimgs(CIFAR10)
train_idxs = 1:49000
train_imgs = [getarray(X[i].img) for i in train_idxs]
train_labels = float.(onehotbatch([X[i].ground_truth.class for i in train_idxs],1:10))
train_set = make_minibatch(train_imgs,train_labels,batch_size)

verify_idxs = 49001:50000
verify_imgs = cat([getarray(X[i].img) for i in verify_idxs]..., dims = 4)
verify_labels = float.(onehotbatch([X[i].ground_truth.class for i in verify_idxs],1:10))
verify_set = [(verify_imgs,verify_labels)]

# Fetch the test data from Metalhead and get it into proper shape.
# CIFAR-10 does not specify a verify set so valimgs fetch the testdata instead of testimgs
tX = valimgs(CIFAR10)
test_idxs = 1:10000
test_imgs = [getarray(tX[i].img) for i in test_idxs]
test_labels = float.(onehotbatch([tX[i].ground_truth.class for i in test_idxs], 1:10))
test_set = make_minibatch(test_imgs,test_labels,batch_size)

# Defining the loss and accuracy functions
loss(x, y) = crossentropy(m(x) .+ ϵ, y)

function accuracy(data_set)
  batch_size = size(data_set[1][1])[end]
  l = length(data_set)*batch_size
  s = 0f0
  for (x,y) in data_set
    s += sum((onecold(m(x|>gpu) |> cpu) .== onecold(y|>cpu)))
  end
  return s/l
end

# Make sure our is nicely precompiled befor starting our training loop
m(train_set[1][1] |> gpu)

# Defining the callback and the optimizer
opt = ADAM(0.001)

@info "Training model..."

for epoch_idx in 1:epochs
  accs = Array{Float32}(undef,0)

  train_set_len = length(train_set)
  shuffle_idxs = collect(1:train_set_len)
  shuffle!(shuffle_idxs)

  for (idx,data_idx) in enumerate(shuffle_idxs)
    (x,y) = train_set[data_idx]
    # We augment `x` a little bit here, adding in random noise
    x = (x .+ ϵ*randn(eltype(x),size(x))) |> gpu
    y = y|> gpu
    Flux.train!(loss,params(m),[(x,y)],opt)
    v_acc = accuracy(verify_set)
    @info "Epoch# $(epoch_idx)/$(epochs) - #$(idx)/$(train_set_len) loss: $(loss(x,y)), accuracy: $(v_acc)"
    push!(accs,v_acc)
  end
  m_acc = mean(accs)
  @info " -> Verify accuracy(mean) : $(m_acc)"
end
test_acc = accuracy(test_set)
@info "Test accuracy : $(test_acc)"
