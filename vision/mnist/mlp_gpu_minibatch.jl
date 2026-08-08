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
const ϵ = 1.0f-32

# Load training labels and images from Flux.Data.MNIST
@info("Loading data...")

train_imgs = MNIST.images()
train_labels = MNIST.labels()

# use 1nd GPU : default
CUDAnative.device!(0)
# use 2nd GPU
#CUDAnative.device!(1)

# Bundle images together with labels and group into minibatch
function make_minibatch(imgs,labels,batch_size)
  X = hcat(float.(reshape.(imgs,:))...) |> gpu
  Y = float.(onehotbatch(labels,0:9)) |> gpu

  data_set = [(X[:,i],Y[:,i]) for i in partition(1:length(labels),batch_size)]
  return data_set
end

@info("Making model...")
# Model
m = Chain(
  Dense(28^2,32,relu),
  Dense(32,10),
  softmax
) |> gpu
loss(x,y) = crossentropy(m(x) .+ ϵ, y)
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
