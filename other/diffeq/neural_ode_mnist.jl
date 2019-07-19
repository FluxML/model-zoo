using OrdinaryDiffEq
using Flux
using Flux: crossentropy
using MLDatasets: MNIST
using MLDataUtils
using DiffEqFlux
using CuArrays
using NNlib

using Base.Iterators: repeated


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
             Conv((3,3),1=>64,relu,stride=1),
             GroupNorm(64,64),
             Conv((4,4),64=>64,relu,stride=2,pad=1),
             GroupNorm(64,64),
             Conv((4,4),64=>64,stride=2,pad=1),
            )|>gpu


nn = Chain(
           Conv((3,3),64=>64,relu,stride=1,pad=1),
           Conv((3,3),64=>64,relu,stride=1,pad=1)
          ) |>gpu

fc = Chain(GroupNorm(64,64),
           x->relu.(x),
           MeanPool((6,6)),
           x -> reshape(x,(64,bs)),
           Dense(64,10,σ)
          )|>gpu


nn_ode(x) = neural_ode(nn,x,(0.f0,1.f0), Tsit5(),save_everystep=false,reltol=1e-2,abstol=1e-2, save_start=false)
m = Chain(down,nn_ode,fc)
m_no_ode = Chain(down,nn,fc)

# Showing this works
x_m = m(x_train[1])
x_m_no_ode = m_no_ode(x_train[1])


function loss(x,y)
    y_hat = m(x)
    return crossentropy(y_hat,y)
end

function loss_no_ode(x,y)
    y_hat = m_no_ode(x)
    return crossentropy(y_hat,y)
end

loss(x_train[1],y_train[1])
loss_no_ode(x_train[1],y_train[1])

opt=ADAM()
iter = 0
cb() = begin
    global iter += 1
    @show iter
    @show loss(x_train[1],y_train[1])
end

Flux.train!(loss_no_ode,params(m_no_ode),zip(x_train,y_train),opt, cb=cb)

Flux.train!(loss,params(m_no_ode),zip(x_train,y_train),opt, cb=cb)

loss((x_train[1],y_train[1])...)

zip(x_train,y_train)

params(Chain(down,nn,fc)) == params(m)
