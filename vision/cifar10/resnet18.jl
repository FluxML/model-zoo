#Should work with
#Julia 1.4
#Atom v0.12.10
#CUDAapi v4.0.0
#CUDAdrv v6.2.2
#CUDAnative v3.0.2
#CuArrays v2.0.1 #master (https://github.com/JuliaGPU/CuArrays.jl)
#DataFrames v0.20.2
#Flux v0.10.3 #master (https://github.com/FluxML/Flux.jl)
#ImageMagick v1.1.3
#Images v0.22.0
#Juno v0.8.1
#MLDatasets v0.4.1
#Metalhead v0.5.0
#NNlib v0.6.6
#RDatasets v0.6.1
#StatsBase v0.32.2
#Zygote v0.4.12
#
#
#
#Still has issues with speed and memory consumption.
#ConvTranspose slows the neural network down significantly.
#A more optimal way to feed images to the neural network probably can be found.
#resizing images in preprocessing should be faster but might lead to greater
#memory consumption. Resizing in preprocessing could be done with ConvTranspose,
#imresize, or by some other tool.


ENV["JULIA_CUDA_VERBOSE"] = true
ENV["JULIA_CUDA_MEMORY_POOL"] = "split"
ENV["JULIA_CUDA_MEMORY_LIMIT"] = 8000_000_000

using Random
using Statistics
using CuArrays
using Zygote
using Flux, Flux.Optimise
using Metalhead, Images
using Metalhead: trainimgs
using Images.ImageCore
using Flux: onehotbatch, onecold, logitcrossentropy, Momentum, @epochs
using Base.Iterators: partition
using Dates

CuArrays.allowscalar(false)

batch_size = 1

getarray(X) = Float32.(permutedims(channelview(X), (2, 3, 1)))

function make_minibatch(imgs,labels,batch_size)
  data_set = [(cat(imgs[i]..., dims = 4), labels[:,i])
          for i in partition(1:length(imgs), batch_size)]
  return data_set
end

X = trainimgs(CIFAR10)

train_idxs = 1:49000

train_imgs = [getarray(X[i].img) for i in train_idxs]
train_labels = Float32.(onehotbatch([X[i].ground_truth.class for i in train_idxs],1:10))
train_set = make_minibatch(train_imgs,train_labels,batch_size)
train_data = train_set |>
  x -> map(y->gpu.(y),x)

verify_idxs = 49001:50000
verify_imgs = cat([getarray(X[i].img) for i in verify_idxs]..., dims = 4)
verify_labels = Float32.(onehotbatch([X[i].ground_truth.class for i in verify_idxs],1:10))
verify_set = [(verify_imgs,verify_labels)]
verify_data = verify_set |>
  x -> map(y->gpu.(y),x)

tX = valimgs(CIFAR10)
test_idxs = 1:10000
test_imgs = [getarray(tX[i].img) for i in test_idxs]
test_labels = Float32.(onehotbatch([tX[i].ground_truth.class for i in test_idxs], 1:10))
test_set = make_minibatch(test_imgs,test_labels,batch_size)
test_data = test_set |>
  x -> map(y->gpu.(y),x)



identity_layer(n) = Chain(Conv((3,3), n=>n, pad = (1,1), stride = (1,1)),
                                  BatchNorm(n,relu),
                                  Conv((3,3), n=>n, pad = (1,1), stride = (1,1)),
                                  BatchNorm(n,relu))

convolution_layer(n) = Chain(Conv((3,3), n=> 2*n, pad = (1,1), stride = (2,2)),
                             BatchNorm(2*n,relu),
                             Conv((3,3), 2*n=>2*n, pad = (1,1), stride = (1,1)),
                             BatchNorm(2*n,relu))

simple_convolution(n) = Chain(Conv((1,1), n=>n, pad = (1,1), stride = (2,2)),
                              BatchNorm(n,relu))


m_filter(n) = Chain(
  Conv((3,3), n=>2*n, pad = (1,1), stride = (2,2)),
  BatchNorm(2*n,relu),
) |> gpu

struct Combinator
    conv::Chain
end |> gpu
Combinator(n) = Combinator(m_filter(n))# |> gpu


function (op::Combinator)(x, y)
  z = op.conv(y)
  return x + z
end


n = 7

m = Chain(
  ConvTranspose((n, n), 3 => 3, stride = n),
  Conv((7,7), 3=>64, pad = (3,3), stride = (2,2)),
  BatchNorm(64,relu),
  MaxPool((3,3), pad = (1,1), stride = (2,2)),
  SkipConnection(identity_layer(64), (variable_1, variable_2) -> variable_1 + variable_2),
  SkipConnection(identity_layer(64), (variable_1, variable_2) -> variable_1 + variable_2),
  SkipConnection(convolution_layer(64), Combinator(64)),
  SkipConnection(identity_layer(128), (variable_1, variable_2) -> variable_1 + variable_2),
  SkipConnection(convolution_layer(128), Combinator(128)),
  SkipConnection(identity_layer(256), (variable_1, variable_2) -> variable_1 + variable_2),
  SkipConnection(convolution_layer(256), Combinator(256)),
  SkipConnection(identity_layer(512), (variable_1, variable_2) -> variable_1 + variable_2),
  MeanPool((7,7)),
  x -> reshape(x, :, size(x,4)),
  Dense(512*1, 10),
  softmax,
) |> gpu



function accuracy(data_set)
  batch_size = size(data_set[1][1])[end]
  l = length(data_set)*batch_size
  s = 0f0
  for (x,y) in data_set
    s += sum((onecold(m(x|>gpu) |> cpu) .== onecold(y|>cpu)))
  end
  return s/l
end


loss(x, y) = sum(logitcrossentropy(m(x), y))
opt = Momentum(0.01)

number_of_epochs = 1

@epochs number_of_epochs train!(loss, params(m), train_data, opt, cb = Flux.throttle(() -> println("training... $(Dates.Time(Dates.now()))") , 10))


verify_acc = accuracy(verify_data)
@info "Verify accuracy : $(verify_acc)"


test_acc = accuracy(test_data)
@info "Test accuracy : $(test_acc)"
