using OrdinaryDiffEq
using Flux
using Flux: logitcrossentropy
using MLDatasets: MNIST
using MLDataUtils
using DiffEqFlux
using CuArrays; CuArrays.allowscalar(false)
using NNlib

function loadmnist(batchsize=bs)
	# Use MLDataUtils LabelEnc for natural onehot conversion
    onehot(labels_raw) = convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:9)))
	# Load MNIST
	imgs, labels_raw = MNIST.traindata();
	# Process images into (H,W,C,BS) batches
	x_train = reshape(imgs,size(imgs,1),size(imgs,2),1,size(imgs,3))|>gpu
	x_train = batchview(x_train,batchsize);
	# Onehot and batch the labels
	y_train = onehot(labels_raw)|>gpu
	y_train = batchview(y_train,batchsize)
	return x_train, y_train
end

# Main
const bs = 128
x_train, y_train = loadmnist(bs);

down = Chain(
             Conv((3,3),1=>64,relu,stride=1),
             GroupNorm(64,64),
             Conv((4,4),64=>64,relu,stride=2,pad=1),
             GroupNorm(64,64),
             Conv((4,4),64=>64,stride=2,pad=1),
            )|>gpu;
nn = Chain(
           #= x->(global nfe+=1;x), =#
           Conv((3,3),64=>64,relu,stride=1,pad=1),
           Conv((3,3),64=>64,relu,stride=1,pad=1)
          ) |>gpu;
fc = Chain(GroupNorm(64,64),
           x->relu.(x),
           MeanPool((6,6)),
           x -> reshape(x,(64,bs)),
           Dense(64,10)
          )|>gpu;

neural_ode_layer = DiffEqFlux.NeuralODE(
                          nn,
                          (0.f0,1.f0),
                          Tsit5(),
                          Dict(
                               :save_everystep=>false,
                               :reltol=>1e-3,
                               :abstol=>1e-3
                              )
                         )



model = Chain(down,neural_ode_layer,fc)

# Showing this works
x_m = model(x_train[1])

function loss(x,y)
    y_hat = model(x)
    return logitcrossentropy(y_hat,y)
end

loss(x_train[1],y_train[1])

classify(x) = argmax.(eachcol(x))
function accuracy(model,data; n_batches=100)
    total_correct = 0
    total = 0
    for (x,y) in collect(data)[1:n_batches]
        target_class = classify(cpu(y))
        predicted_class = classify(cpu(model(x)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct/total
end

accuracy(model, zip(x_train,y_train))

opt=ADAM()
iter = 0
cb() = begin
    global iter += 1
    @show iter
    @show nfe
    @show loss(x_train[1],y_train[1])
    @show no.model[2].weight[1]
    if iter%10 == 0
        @show accuracy(m, zip(x_train,y_train))
    end
    global nfe=0
end


Flux.train!(loss,params(model),zip(x_train,y_train),opt, cb=cb)

Flux.Tracker.gradient(()->loss(x_train[1],y_train[1]),params(m_no))


# Saving doesn't work yet
using BSON: @save, @load

m_cpu = cpu(m)
@save "saved_models/mnist_node.bson" m_cpu

@load "saved_models/mnist_node.bson" m_cpu



