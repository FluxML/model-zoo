# Ref: http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/
using Images
using Flux
using Parameters: @with_kw

# set parameters
@with_kw mutable struct Args
	z_dim::Int = 2		# Dim of Latent Vector
	x_dim::Int = 512	# X-Dimension of Image
	y_dim::Int = 512	# Y-Dimension of Image
	N::Int = 14			#
	hidden::Int = 9		# Number of hidden layers in the image
	batch_size::Int = 1024	# Batch Size for prediction
end

# cast 0:x-1 to -0.5:0.5
cast(x) = [range(-0.5, stop=0.5, step=1/(x - 1))...]

function getData(args)
    xs, ys = cast(args.x_dim), cast(args.y_dim)
    xs = repeat(xs, inner=(args.y_dim))
    ys = repeat(ys, outer=(args.x_dim))
	# Radius term for each point of input
    rs = sqrt.(xs.^2 + ys.^2)
	return xs,ys,rs
end

#Definition for each individual layer
# sample weigths from a gaussian distribution
unit(args, in=args.N, out=args.N, f=tanh) = Dense(in, out, f, initW=randn)

function Construct_model(args)
    # input -> [x, y, r, z...]
    layers = Any[unit(args, 3 + args.z_dim)]
    for i=1:args.hidden
        push!(layers, unit(args))
    end
    push!(layers, unit(args,args.N, 1, σ))
    model = Chain(layers...)
    return model
end

function batch(arr, s)
    batches = []
    l = size(arr, 2)
    for i=1:s:l
        push!(batches, arr[:, i:min(i+s-1, l)])
    end
    batches
end

function getImage(z, model, args) 
    n = args.x_dim * args.y_dim   
    z = repeat(reshape(z, 1, args.z_dim), outer=(n, 1))
	xs, ys, rs = getData(args)
    coords = hcat(xs, ys, rs, z)'
    
    coords = batch(coords, args.batch_size)
    
	# Pixel value at a position x is defined as output of model at that point 
    getColorAt(x) = model(x)

    pixels = [Gray.(hcat(getColorAt.(coords)...))...]
    reshape(pixels, args.y_dim, args.x_dim)
end

function saveImg(z, model, args, image_path=joinpath(dirname(@__FILE__),"sample.png"))
    imgg = getImage(z, model, args)
    save(image_path, imgg)
    imgg
end

function generateImg(; kws...)
	args = Args(; kws...)
	
	model = Construct_model(args)
	
	#Saving image as "sample.png"
	saveImg(rand(args.z_dim), model, args)
end

cd(@__DIR__)
generateImg()
