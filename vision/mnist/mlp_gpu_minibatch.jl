#=
Julia version: 1.3.1
Flux version : 0.10.1
=#
__precompile__()
module MNIST_BATCH
using Flux
using Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy,throttle
using Base.Iterators: repeated,partition

using CUDAnative
using CuArrays
CuArrays.allowscalar(false)

#= 
Very important !!
ϵ is used to prevent loss NaN
=#
const ϵ = 1.0f-10

# Load training labels and images from Flux.Data.MNIST
@info("Loading data...")
#=
MNIST.images() : [(28x28),...,(28x28)] 60,000x28x28 training images
MNIST.labels() : 0 ~ 9 labels , 60,000x10 training labels
=#
train_imgs = MNIST.images()
train_labels = MNIST.labels()

# use 1nd GPU : default
#CUDAnative.device!(0)
# use 2nd GPU
#CUDAnative.device!(1)

# Bundle images together with labels and group into minibatch
function make_minibatch(imgs,labels,batch_size)
  #=
   reshape.(MNIST.images(),:) : [(784,),(784,),...,(784,)]  60,000개의 데이터
   X : (784x60,000)
   Y : (10x60,000)
  =#
  X = hcat(float.(reshape.(imgs,:))...) |> gpu
  Y = float.(onehotbatch(labels,0:9)) |> gpu
  # Y = Float32.(onehotbatch(labels,0:9))
  
  data_set = [(X[:,i],Y[:,i]) for i in partition(1:length(labels),batch_size)]
  return data_set
end

@info("Making model...")
# Model
m = Chain(
  Dense(28^2,32,relu), # y1 = relu(W1*x + b1), y1 : (32x?), W1 : (32x784), b1 : (32,)
  Dense(32,10), # y2 = W2*y1 + b2, y2 : (10,?), W2: (10x32), b2:(10,)
  softmax
) |> gpu
loss(x,y) = crossentropy(m(x) .+ ϵ, y .+ ϵ)
accuracy(x,y) = mean(onecold(m(x)|>cpu) .== onecold(y|>cpu))

batch_size = 500
train_dataset = make_minibatch(train_imgs,train_labels,batch_size)

opt = ADAM()


@info("Training model...")

epochs = 200
# used for plots
accs = Array{Float32}(undef,0)

dataset_len = length(train_dataset)
for i in 1:epochs
  for (idx,dataset) in enumerate(train_dataset)
    Flux.train!(loss,params(m),[dataset],opt)
    # Flux.train!(loss,params(m),[dataset],opt,cb = throttle(()->@show(loss(dataset...)),20))
    acc = accuracy(dataset...)
    if idx == dataset_len
      @info("Epoch# $(i)/$(epochs) - loss: $(loss(dataset...)), accuracy: $(acc)")
      push!(accs,acc)
    end
  end
end

# Test Accuracy
tX = hcat(float.(reshape.(MNIST.images(:test),:))...) |> gpu
tY = float.(onehotbatch(MNIST.labels(:test),0:9)) |> gpu

println("Test loss:", loss(tX,tY))
println("Test accuracy:", accuracy(tX,tY))

end

using Plots;gr()
plot(MNIST_BATCH.accs)

