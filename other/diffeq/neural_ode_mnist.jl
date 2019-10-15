using OrdinaryDiffEq
using Flux
using Flux: logitcrossentropy
using MLDatasets: MNIST
using MLDataUtils
using DiffEqFlux
using CuArrays
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
x_train, y_train = loadmnist(bs)

down = Chain(
             x->reshape(x,(28*28,:)),
             Dense(784,20,tanh)
            )|>gpu
nfe=0
nn = Chain(
           #= x->(global nfe+=1;x), =#
           Dense(20,10,tanh),
           Dense(10,10,tanh),
           Dense(10,20,tanh)
          ) |>gpu
fc = Chain(
           Dense(20,10)
          )|>gpu

CuArrays.allowscalar(false)
nn_ode(x) = neural_ode(nn,x,gpu((0.f0,1.f0)), Tsit5(),save_everystep=false,reltol=1e-3,abstol=1e-3, save_start=false)
m = Chain(down,nn_ode,fc)
m_no_ode = Chain(down,nn,fc)

x_d = down(x_train[1])
nn_ode(x_d)

## Stop here

# Showing this works
x_m = m(x_train[1])

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

accuracy(m, zip(x_train,y_train))

        
function loss(x,y)
    y_hat = m(x)
    return logitcrossentropy(y_hat,y)
end

loss(x_train[1],y_train[1])

opt=ADAM()
iter = 0
cb() = begin
    global iter += 1
    @show iter
    @show nfe
    @show loss(x_train[1],y_train[1])
    @show cpu(down)[2].W[1]
    if iter%10 == 0
        @show accuracy(m, zip(x_train,y_train))
    end
    global nfe=0
end


Flux.train!(loss,params(down,nn,fc),zip(x_train,y_train),opt, cb=cb)


# Saving doesn't work yet
using BSON: @save, @load

m_cpu = cpu(m)
@save "saved_models/mnist_node.bson" m_cpu

@load "saved_models/mnist_node.bson" m_cpu



