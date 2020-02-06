#=
 cifar10 dataset spec
 - 60,000 images of 32x32 size 
 - train images : 50,000
 - test images : 10,000
 - classify item : 10
 - each class have 6,000 images and 5,000 train images, 1,000 test images
 
 Data format:
 WHCN order : (width, height, #channels, #batches)
 ex) A single 100x100 RGB image data format : 100x100x3x1
 =#

# Julia version : 1.3.1
# Flux version : v0.10.1

__precompile__()
module _CIFAR10
using Flux, Metalhead, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Metalhead: trainimgs
using Images: channelview
using Statistics: mean
using Base.Iterators: partition
using CUDAnative
using CuArrays
CuArrays.allowscalar(false)

using BSON: @save
using Logging
using Dates

const model_file = "./cifar10_vgg16_model.bson"
const log_file ="./cifar10_vgg16.log"

# Very important : this prevent loss NaN
const ϵ = 1.0f-10

# use 1nd GPU : default
#CUDAnative.device!(0)
# use 2nd GPU
#CUDAnative.device!(1)

log = open(log_file, "w+")
global_logger(ConsoleLogger(log))

@info "Start - $(now())"
@info "Config VGG16, VGG19 models ..."
flush(log)
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
  softmax) |> gpu

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
  softmax) |> gpu

# Function to convert the RGB image to Float32 Arrays
getarray(X) = Float32.(permutedims(channelview(X), (2, 3, 1)))


@info "Data download and preparing ..."

function make_minibatch(imgs,labels,batch_size)
  data_set = [(cat(imgs[i]..., dims = 4) |> gpu, 
          labels[:,i]) |> gpu 
          for i in partition(1:length(imgs), batch_size)]
  return data_set
end

epochs = 40
batch_size = 100

X = trainimgs(CIFAR10)

train_idxs = 1:49000
train_imgs = [getarray(X[i].img) for i in train_idxs]
train_labels = float.(onehotbatch([X[i].ground_truth.class for i in train_idxs],1:10))
train_dataset = make_minibatch(train_imgs,train_labels,batch_size)

valid_idxs = 49001:50000
valX = cat([getarray(X[i].img) for i in valid_idxs]..., dims = 4) |> gpu
valY = float.(onehotbatch([X[i].ground_truth.class for i in valid_idxs],1:10)) |> gpu

# Defining the loss and accuracy functions

@info "VGG16 models instantiation ..."
m = vgg16()

loss(x, y) = crossentropy(m(x) .+ ϵ, y .+ ϵ)

accuracy(x, y) = mean(onecold(m(x)|>cpu, 1:10) .== onecold(y|>cpu, 1:10))

# Defining the callback and the optimizer

evalcb = throttle(() -> @info(accuracy(valX, valY)), 10)

opt = ADAM()

@info "Training model..."


@time begin
dataset_len = length(train_dataset)
for i in 1:epochs
  for (idx,dataset) in enumerate(train_dataset)
    Flux.train!(loss,params(m),[dataset],opt)
    #Flux.train!(loss,params(m),[dataset],opt,cb = evalcb)    
    acc = accuracy(valX,valY)
    @info "Epoch# $(i)/$(epochs) - #$(idx)/$(dataset_len) loss: $(loss(dataset...)), accuracy: $(acc)"
    flush(log)
  end
  @save model_file m
end
end # end of @time

# Fetch the test data from Metalhead and get it into proper shape.
# CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs
tX = valimgs(CIFAR10)
test_idxs = 1:10000
test_imgs = [getarray(tX[i].img) for i in test_idxs]
test_labels = float.(onehotbatch([tX[i].ground_truth.class for i in test_idxs], 1:10))
test_dataset = make_minibatch(test_imgs,test_labels,batch_size)

dataset_len = length(test_dataset)
for (idx,dataset) in enumerate(test_dataset)
  acc = accuracy(dataset...)
end

@info "Test accuracy : $(mean(test_accs))"
@info "End - $(now())"
close(log)
end
