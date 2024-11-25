using DelimitedFiles
using Images
using JLD2, FileIO
using Metalhead
using Flux: onehotbatch, onecold, crossentropy, throttle, Optimise
using Statistics
using Flux
using Flux: @epochs
#using CuArrays
"""
This is an example of transfer learning in which we learn Tiny Imagenet dataset using Metalhead's Resnet
About tiny imagenet:
much like Imagenet, the tiny Imagenet consists of image data of various categories.
It has 200 different classes and 100,000 training examples
It can be downloaded from here: https://tiny-imagenet.herokuapp.com/ 
"""
function create_minibatch(batch_num, mini_batch_size)
	"""
	Function that returns a minibatch tuple for a given batch number based of mini batch size
	"""
	batch_x = zeros(2048, mini_batch_size)
	batch_y = zeros(mini_batch_size)
	
	classes = readdlm("tiny-imagenet-200/wnids.txt", '\n', String)
	
	base = 0

	for s in classes
		data = zeros(64, 64, 3, mini_batch_size ÷ 200)
		imgsloc = "tiny-imagenet-200/train/$s/images"

		i_begin = (batch_num-1)*(mini_batch_size ÷ 200) + 1
		i_end = batch_num*(mini_batch_size ÷ 200)
		label = base ÷ (mini_batch_size ÷ 200)
		for i in  i_begin:i_end
			img = load("$imgsloc/$(s)_$(i-1).JPEG")
			if typeof(img) == Array{Gray{Normed{UInt8,8}},2} #Some of the images are gray so c them to RGB
				img = colorview(RGB, img, img ,img)
			end
			img = channelview(img)
			img = permutedims(img, [2, 3, 1]) #WHCN
			data[:, :, :, i - i_begin + 1] = img
			batch_y[base + i - i_begin + 1] =  label
		end
		#Save the features from ResNet forward props into the batches
		batch_x[:, base+1:base+mini_batch_size ÷ 200] = Tracker.data(tinyResNet[1](data))
		base = base + mini_batch_size ÷ 200
	end
	batch_y = onehotbatch(batch_y, 0:199)
	return (batch_x, batch_y);
end

function load_val_set()
	"""
	Returns val_x, val_y
	"""
	val_x = zeros(2048, 10000)
	val_y = zeros(10000)

	classes = readdlm("tiny-imagenet-200/wnids.txt", '\n', String)

	#Creating a dictionary of classnames to classnumber for tinyImagenet
	class_dict = Dict()
	class_num = 0
	for s in classes
		push!(class_dict, s => class_num)
		class_num += 1
	end

	imgsloc = "tiny-imagenet-200/val/images"
	data = zeros(64, 64, 3, 100);
	for i in 1:10000
		img = load("$imgsloc/val_$(i-1).JPEG")
		if typeof(img) == Array{Gray{Normed{UInt8,8}},2} #Some of the images are gray so c them to RGB
			img = colorview(RGB, img, img ,img)
		end
		img = channelview(img)
		img = permutedims(img, [2, 3, 1]) #WHCN
		if i%100 == 0
			data[:, :, :, 100] = img
			val_x[:, i-99:i] = Tracker.data(tinyResNet[1](data));
		else
			data[:, :, :, i%100] = img
		end
	end
	entries = readdlm("tiny-imagenet-200/val/val_annotations.txt", '\n', String)
	i = 1
	for e in entries
		val_y[i] = class_dict[match(r"JPEG.*", e).match[6:14]]
	end
	val_y = onehotbatch(val_y, 0:199)
	return (val_x, val_y);
end


function model()
	"""
	Returns modified ResNet
	The model is made of two different chains:
	First chain consists on All but last two layers of ResNet 
	Second chain is the one we will train
	"""
	fixed = ResNet().layers[1:end-2]
	trainable = Chain(
				Dense(2048, 200),
				softmax)
	m = Chain(fixed, trainable) |> gpu
	return m
end

#create model instance
tinyResNet = model()

#make mini batches
mini_batch_size = 20000
Dataset = []
println("Creating minibatches of size $mini_batch_size (This may take a while)")
for batch_num in 1:100000÷mini_batch_size
    println("Creating minibatch: $batch_num")
    push!(Dataset, (create_minibatch(batch_num, mini_batch_size)));
end

Dataset = Dataset |> gpu

#Save the minibatches because these are now the features to train the final layer of custom ResNet
#The file size will be around 1.6 GB
#@save "minibatches.jld2" Dataset

#Load the minibatches if they're not in your workspace
#@load "minibatches.jld2" Dataset
#Now train trainable part only
loss(x, y) = crossentropy(tinyResNet[2](x), y)
accuracy(x, y) = mean(onecold(tinyResNet[2](x)) .== onecold(y))
opt = ADAM(0.1)
evalcb = throttle(() -> @show(loss(Dataset[1][1], Dataset[1][2])), 10)
@epochs 10 Flux.train!(loss, params(tinyResNet[2]), Dataset, opt, cb = evalcb)

#@save "model.jld2" tinyResNet
#@load "model.jld2" tinyResnet

#Calculate training accuracy accross all the minibatches
avrg = 0
for i in 1: 100000÷mini_batch_size
	global avrg += accuracy(Dataset[i][1], Dataset[i][2])
end
avrg /= 100000÷mini_batch_size

println("training accuracy: ", avrg)

println("Loading validation set (This may take a while)")
Valset = load_val_set() |> gpu

println("validation accuracy: ", accuracy(Valset[1], Valset[2]))
#@load "Valset.jld2" Valset