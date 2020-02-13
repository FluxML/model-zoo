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

working_path = dirname(@__FILE__)
file_path(file_name) = joinpath(working_path,file_name)
include(file_path("cmd_parser.jl"))

model_file = file_path("cifar10_vgg16_model.bson")

# Get arguments
parsed_args = CmdParser.parse_commandline()

epochs = parsed_args["epochs"]
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

# Very important : this prevent loss NaN
ϵ = 1.0f-10


# use 1nd GPU : default
#CUDAnative.device!(0)
device!(gpu_device)
CuArrays.allowscalar(false)

@info "Config VGG16, VGG19 models ...";flush(log)

acc = 0; epoch = 0
if use_saved_model && isfile(model_file) && filesize(model_file) > 0
  # flush : 버퍼링 없이 즉각 log를 파일 또는 console에 write하도록 함
  @info "Load saved model $(model_file) ...";flush(log)
  # model : @save시 사용한 object명
  @load model_file model acc epoch  
  m = model |> gpu
  @info " -> accuracy : $(acc), epochs : $(epoch)";flush(log)
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
  data_set = [(cat(imgs[i]..., dims = 4), 
          labels[:,i]) 
          for i in partition(1:length(imgs), batch_size)]
  return data_set
end
# Fetching the train and verify data and getting them into proper shape
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

@info "VGG16 models instantiation ...";flush(log)

loss(x, y) = crossentropy(m(x) .+ ϵ, y .+ ϵ)

# accuracy(x, y) = mean(onecold(m(x)|>cpu, 1:10) .== onecold(y|>cpu, 1:10))
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
# train_set[1][1] : (28,28,1,batch_size)
@info "Model pre-compile...";flush(log)
m(train_set[1][1] |> gpu)

# Defining the callback and the optimizer
# evalcb = throttle(() -> @info(accuracy(verify_set)), 10)
opt = ADAM(0.001)

@info "Training model...";flush(log)
best_acc = acc
last_improvement = epoch
# used for plots
for epoch_idx in 1+epoch:(epochs+=epoch)
  accs = Array{Float32}(undef,0)
  global best_acc, last_improvement
  train_set_len = length(train_set)
  shuffle_idxs = collect(1:train_set_len)
  shuffle!(shuffle_idxs)  

  for (idx,data_idx) in enumerate(shuffle_idxs)
    (x,y) = train_set[data_idx]
    # We augment `x` a little bit here, adding in random noise
    x = (x .+ 0.1f0*randn(eltype(x),size(x))) |> gpu
    y = y|> gpu    
    Flux.train!(loss,params(m),[(x,y)],opt)
    #Flux.train!(loss,params(m),[(x,y)],opt,cb = evalcb)    
    v_acc = accuracy(verify_set)
    @info "Epoch# $(epoch_idx)/$(epochs) - #$(idx)/$(train_set_len) loss: $(loss(x,y)), accuracy: $(v_acc)";flush(log)
    # @info "Epoch# $(epoch_idx)/$(epochs) - #$(idx)/$(train_set_len) accuracy: $(v_acc)";flush(log)
    push!(accs,v_acc)
  end # for

  m_acc = mean(accs)
  @info " -> Verify accuracy(mean) : $(m_acc)";flush(log)
  test_acc = accuracy(test_set)
  @info "Test accuracy : $(test_acc)";flush(log)  
  
  # If our accuracy is good enough, quit out.
  if test_acc >= 0.98
    @info " -> Early-exiting: We reached our target accuracy of 98%";flush(log)
    model = m |> cpu;acc = test_acc;epoch = epoch_idx
    @save model_file model acc epoch
    break
  end
  
  # If this is the best accuracy we've seen so far, save the model out
  if test_acc >= best_acc
    @info " -> New best accuracy! saving model out to $(model_file)"; flush(log)
    model = m |> cpu;acc = test_acc;epoch = epoch_idx
    # @save,@load 시 같은 이름을 사용해야 함, 여기서는 "model"을 사용함
    @save model_file model acc epoch
    best_acc = test_acc
    last_improvement = epoch_idx    
  end
  
  # If we haven't seen improvement in 5 epochs, drop out learning rate:
  if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
    opt.eta /= 10.0
    @info " -> Haven't improved in a while, dropping learning rate to $(opt.eta)!";flush(log)
    # After dropping learning rate, give it a  few epochs to improve
    last_improvement = epoch_idx
  end  
  
  if epoch_idx - last_improvement >= 10  
    @info " -> We're calling this converged."; flush(log)
    break
  end
end # end of for

@info "End - $(now())"
if create_log_file
  close(log)
end

