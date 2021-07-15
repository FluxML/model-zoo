using Images, Augmentor, Flux
using Flux: onehotbatch, argmax, crossentropy, throttle, @epochs
using Base.Iterators: repeated, partition
using BSON: @save, @load
# using CuArrays
using BinDeps

if(isfile("scenes.zip"))
	download("http://cvcl.mit.edu/MM/downloads/Scenes.zip", "scenes.zip")

	run(unpack_cmd("./scenes.zip", "./Scenes Dataset", ".zip", ""))

	isfile("scenes.zip") || rm("scenes.zip")
end

run(`find -type f -name ".*" -delete`)

include("augment.jl")
include("load_data.jl")
include("resnet.jl")
include("train.jl")
