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
__precompile__(true)
using Random
using BSON
using BSON: @save,@load
using Logging
using Dates
using NNlib 
using CuArrays
using CUDAdrv
using CUDAnative: device!
using Flux, Metalhead, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Metalhead: trainimgs
using Images: channelview
using Statistics: mean
using Base.Iterators: partition
using ArgParse
#=
Argument parsing 
=#

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--epoch","-e"
            help = "epoch number, default=30"
            arg_type = Int
            default = 30
        "--batch", "-b"
            help = "mini-batch size, default=200"
            arg_type = Int
            default = 100
        "--gpu", "-g"
            help = "gpu index to use , 0,1,2,3,.., default=0"
            arg_type = Int
            default = 0
        "--model", "-m"
            help = "use saved model file"
            arg_type = Bool
            default = true
        "--log","-l"
            help = "create log file"
            arg_type = Bool
            default = true            
    end

    return parse_args(s)
end
parsed_args = parse_commandline()

epochs = parsed_args["epoch"]
batch_size = parsed_args["batch"]
use_saved_model = parsed_args["model"]
gpu_device = parsed_args["gpu"]
create_log_file = parsed_args["log"]

if create_log_file
  log_file ="./cifar10_vgg16_$(Dates.format(now(),"yyyymmdd-HHMMSS")).log"
  log = open(log_file, "w+")
else
  log = stdout
end
global_logger(ConsoleLogger(log))

@info "Start - $(now())";flush(log)


@info "=============== Arguments ==============="
@info "epochs=$(epochs)"
@info "batch_size=$(batch_size)"
@info "use_saved_model=$(use_saved_model)"
@info "gpu_device=$(gpu_device)"
@info "=========================================";flush(log)

const model_file = "./cifar10_vgg16_model.bson"

# Very important : this prevent loss NaN
const ϵ = 1.0f-10


# use 1nd GPU : default
#CUDAnative.device!(0)
device!(gpu_device)
CuArrays.allowscalar(false)

@info "Config VGG16, VGG19 models ...";flush(log)

if use_saved_model && isfile(model_file)
  # flush : 버퍼링 없이 즉각 log를 파일 또는 console에 write하도록 함
  @info "Load saved model $(model_file) ...";flush(log)
  # model : @save시 사용한 object명
  @load model_file model
  m = model |> gpu
else
  @info "Create new model ...";flush(log)
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
end
# 
# Function to convert the RGB image to Float64 Arrays
#=
1)channelview로 이미지의 color를 channel별로 분리한다.
- 분리된 channel은 맨앞에 새로운 차원을 추가 하여 channel을 분리한다.
- 예) 32x32 이미지의 채널을 분리하면 3x32x32로 3개의 채널이 추가 된다
2)permutedims로 분리된 채널을 뒤로 보낸다.
- Flux에서 사용되는 이미지 포맷은 WHCN-width,height,#channel,#batches 이다
- 채널분리된 이미지가 3x32x32인 경우 permutedims(img,(2,3,1))을 적용하면
- 32x32x3으로 width,height,#channel 순으로 바뀐다.
=#
getarray(X) = Float32.(permutedims(channelview(X), (2, 3, 1)))


@info "Data download and preparing ...";flush(log)
function make_minibatch(imgs,labels,batch_size)
  data_set = [(cat(imgs[i]..., dims = 4) |> gpu, 
          labels[:,i]) |> gpu 
          for i in partition(1:length(imgs), batch_size)]
  return data_set
end
# Fetching the train and validation data and getting them into proper shape
#=
trainimgs(모듈명) : 
 - 모듈명이 들어 가면 모듈명에 관련된 train용 데이터를 다운받아 리턴한다.
 - ex) trainimgs(CIFAR10) : 50,000개의 train data가 return 된다.
X 
=#

X = trainimgs(CIFAR10)
# Training용 데이터 준비
# 이미지 채널 분리 및 재배열, training용으로 60,000개중 50,000개를 사용한다.
train_idxs = 1:49000
train_imgs = [getarray(X[i].img) for i in train_idxs]
train_labels = float.(onehotbatch([X[i].ground_truth.class for i in train_idxs],1:10))
train_dataset = make_minibatch(train_imgs,train_labels,batch_size)

valid_idxs = 49001:50000
valX = cat([getarray(X[i].img) for i in valid_idxs]..., dims = 4) |> gpu
valY = float.(onehotbatch([X[i].ground_truth.class for i in valid_idxs],1:10)) |> gpu

# Defining the loss and accuracy functions

@info "VGG16 models instantiation ...";flush(log)

loss(x, y) = crossentropy(m(x) .+ ϵ, y .+ ϵ)

accuracy(x, y) = mean(onecold(m(x)|>cpu, 1:10) .== onecold(y|>cpu, 1:10))

# Defining the callback and the optimizer

evalcb = throttle(() -> @info(accuracy(valX, valY)), 10)

opt = ADAM()

@info "Training model...";flush(log)

# used for plots
# accs = Array{Float32}(undef,0)
@time begin
dataset_len = length(train_dataset)
shuffle_idxs = collect(1:dataset_len)
shuffle!(shuffle_idxs)
for i in 1:epochs
  for (idx,data_idx) in enumerate(shuffle_idxs)
    dataset = train_dataset[data_idx]
    Flux.train!(loss,params(m),[dataset],opt)
    #Flux.train!(loss,params(m),[dataset],opt,cb = evalcb)    
    acc = accuracy(valX,valY)
    @info "Epoch# $(i)/$(epochs) - #$(idx)/$(dataset_len) loss: $(loss(dataset...)), accuracy: $(acc)";flush(log)
    # push!(accs,acc)
  end
  model = m |> cpu  
  # @load 시 여기에서 사용한 "model" 로 로딩 해야 함
  @save model_file model
end
end # end of @time
# Fetch the test data from Metalhead and get it into proper shape.
# CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs
tX = valimgs(CIFAR10)
test_idxs = 1:10000
test_imgs = [getarray(tX[i].img) for i in test_idxs]
test_labels = float.(onehotbatch([tX[i].ground_truth.class for i in test_idxs], 1:10))
test_dataset = make_minibatch(test_imgs,test_labels,batch_size)

test_accs = Array{Float32}(undef,0)
dataset_len = length(test_dataset)
for (idx,dataset) in enumerate(test_dataset)
  acc = accuracy(dataset...)
  push!(test_accs,acc)
end
@info "Test accuracy : $(mean(test_accs))"
@info "End - $(now())"
close(log)

