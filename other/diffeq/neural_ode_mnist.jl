using OrdinaryDiffEq
using Flux
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
	y_train = onehot(labels_raw)
	y_train = batchview(y_train,batchsize)

	return x_train, y_train
end

# Main
const bs = 200
x_train, y_train = loadmnist();

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
           x -> MeanPool(x,PoolDims(x,6))
           Dense(64,10,σ)
          )|>gpu

maxpool(nn(down(x_train[1])),(1,1))

nn(down(x_train[1])) |>size

meanpool(nn(down(x_train[1])),PoolDims(nn(down(x_train[1])),6))|>size

nn(down(x_train[1])) |>size
