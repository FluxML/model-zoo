#=
Test Environment
 - Julia : v1.3.1
 - Flux  : v0.10.1
 Usage:
 - julia conv_gpu_minibatch.jl  --help
 - ex) julia conv_gpu_minibatch.jl -e 100 -b 1000 -g 0 -l false
 -     epochs : 100, batch size: 1000, gpu device index : 0 , log file : false
=#

# Classifies MNIST digits with a convolution network.
# Writes out saved model to the file "mnist_conv.bson".
# Demonstrates basic model construction, training, saving,
# conditional early-exits, and learning rate scheduling.
#
# This model, while simple, should hit around 99% test
# accuracy after training for approximately 20 epochs.

using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Printf, BSON
using Logging
using Dates
using CUDAnative: device!
using CuArrays
using Random
using Dates

working_path = dirname(@__FILE__)
file_path(file_name) = joinpath(working_path,file_name)
include(file_path("cmd_parser.jl"))

model_file = file_path("conv_gpu_minibatch.bson")

# Get arguments
parsed_args = CmdParser.parse_commandline()

epochs = parsed_args["epochs"]
batch_size = parsed_args["batch"]
use_saved_model = parsed_args["model"]
gpu_device = parsed_args["gpu"]
create_log_file = parsed_args["log"]

if create_log_file
    log_file = file_path("conv_gpu_minibatch_$(Dates.format(now(),"yyyymmdd-HHMMSS")).log")
    log = open(log_file,"w+")
else
    log = stdout
end
global_logger(ConsoleLogger(log))

@info "Start - $(now())";flush(log)

@info "============= Arguments ============="
@info "epochs=$(epochs)"
@info "batch_size=$(batch_size)"
@info "use_saved_model=$(use_saved_model)"
@info "gpu_device=$(gpu_device)"
@info "create_log_file=$(create_log_file)"
@info "=====================================";flush(log)

# set using GPU device
device!(gpu_device)
CuArrays.allowscalar(false)

# Load labels and images from Flux.Data.MNIST
@info "Loading data set";flush(log)

# Bundle images together with labels and groups into minibatch
function make_minibatch(imgs,labels,batch_size)
    # WHCN: width x height x #channel x #batch
    # transform (28x28) to (28x28x1x#bacth)
    len = length(imgs)
    sz = size(imgs[1])
    data_set = 
    [(cat([reshape(Float32.(imgs[i]),sz...,1,1) for i in idx]...,dims=4),
      float.(onehotbatch(labels[idx],0:9)) ) for idx in partition(1:len,batch_size) ]
    return data_set
end

# Train data load
# 60,000 labels
train_labels = MNIST.labels()
# 60,000 images : ((28x28),...,(28x28))
train_imgs = MNIST.images()
# Make train data to minibatch
train_set = make_minibatch(train_imgs,train_labels,batch_size)

# Test data load
test_labels = MNIST.labels(:test)
test_imgs = MNIST.images(:test)
test_set = make_minibatch(test_imgs,test_labels,batch_size)

#=
 Define our model. We will use a simple convolutional architecture with
 three iterations of Conv -> ReLu -> MaxPool, followed by a final Dense
 layer that feeds into a softmax probability output.
=#
@info "Construncting model...";flush(log)
model = Chain(
  # First convolution, operating upon a 28x28 image
  Conv((3,3), 1=>16, pad=(1,1), relu),
  MaxPool((2,2)),

  # Second convolution, operating upon a 14x14 image
  Conv((3,3), 16=>32, pad=(1,1), relu),
  MaxPool((2,2)),
  
  # Third convolution, operating upon a 7x7 image
  Conv((3,3), 32=>32, pad=(1,1), relu),
  MaxPool((2,2)),
  
  # Reshape 3d tensor into a 2d one, at this point it should be (3,3,32,N)
  # which is where we get the 288 in the `Dense` layer below:
  x -> reshape(x, :, size(x,4)),
  Dense(288,10),
  
  # Finally, softmax to get nice probabilities
  softmax,
)

m = model |> gpu

#= 
`loss()` calculates the crossentropy loss between our prediction `y_hat`
 (calculated from `m(x)`) and the ground truth `y`. We augment the data
 a bit, adding gaussian random noise to our image to make it more robust.
 =#
function loss(x,y)
 ŷ = m(x)
 return crossentropy(ŷ,y)
end
# Make sure our model is nicely precompiled befor starting our training loop

function accuracy(data_set) 
  l = length(data_set)*batch_size
  s = 0f0
  for (x,y) in data_set
    s += sum((onecold(m(x|>gpu) |> cpu) .== onecold(y|>cpu)))
  end
  return s/l
end

# Make sure our is nicely precompiled befor starting our training loop
# train_set[1][1] : (28,28,1,batch_size)
m(train_set[1][1] |> gpu)

# Train our model with the given training set using the ADAM optimizer and
# printing out performance aganin the test set as we go.
opt = ADAM(0.001)

@info "Beginning training loop...";flush(log)
best_acc = 0.0
last_improvement = 0

@time begin
for epoch_idx in 1:epochs
  global best_acc, last_improvement  
  suffle_idxs = collect(1:length(train_set))
  shuffle!(suffle_idxs)
  for idx in suffle_idxs
    (x,y) = train_set[idx]
    # We augment `x` a little bit here, adding in random noise
    x = (x .+ 0.1f0*randn(eltype(x),size(x))) |> gpu
    y = y|> gpu
    Flux.train!(loss, params(m), [(x, y)],opt)
  end
  acc = accuracy(test_set)
  @info(@sprintf("[%d]: Test accuracy: %.4f",epoch_idx,acc));flush(log)

  # If our accuracy is good enough, quit out.
  if acc >= 0.999
    @info " -> Early-exiting: We reached our target accuracy of 99.9%";flush(log)
    break
  end

  # If this is the best accuracy we've seen so far, save the model out
  if acc >= best_acc
    @info " -> New best accuracy! saving model out to $(model_file)"; flush(log)
    model = m |> cpu
    acc = acc |> cpu
    BSON.@save model_file model epoch_idx acc
    best_acc = acc
    last_improvement = epoch_idx
  end

  #If we haven't seen improvement in 5 epochs, drop out learing rate:
  if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
    opt.eta /= 10.0
    @warn " -> Haven't improved in a while, dropping learning rate to $(opt.eta)!"; flush(log)

    # After dropping learing rate, give it a few epochs to improve
    last_improvement = epoch_idx
  end

  if epoch_idx - last_improvement >= 10
    @warn " -> We're calling this converged.";flush(log)
    break
  end  
end # for
end # @time
@info "End - $(now())"
if create_log_file
  close(log)
end

