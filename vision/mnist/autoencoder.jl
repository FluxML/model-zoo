# Encode MNIST images as compressed vectors that can later be decoded back into
# images.
using Flux, Flux.Data.MNIST
using Flux: @epochs, onehotbatch, mse, throttle
using Base.Iterators: partition
using Parameters: @with_kw
using CUDAapi
if has_cuda()
    @info "CUDA is on"
    import CUDA
    CUDA.allowscalar(false)
end

@with_kw mutable struct Args
    lr::Float64 = 1e-3		# Learning rate
    epochs::Int = 10		# Number of epochs
    N::Int = 32			# Size of the encoding
    batchsize::Int = 1000	# Batch size for training
    sample_len::Int = 20 	# Number of random digits in the sample image
    throttle::Int = 5		# Throttle timeout
end

function get_processed_data(args)
    # Loading Images
    imgs = MNIST.images()
    #Converting image of type RGB to float 
    imgs = channelview.(imgs)
    # Partition into batches of size 1000
    train_data = [float(hcat(vec.(imgs)...)) for imgs in partition(imgs, args.batchsize)]
    
    train_data = gpu.(train_data)
    return train_data
end

function train(; kws...)
    args = Args(; kws...)	

    train_data = get_processed_data(args)

    @info("Constructing model......")
    # You can try to make the encoder/decoder network larger
    # Also, the output of encoder is a coding of the given input.
    # In this case, the input dimension is 28^2 and the output dimension of
    # encoder is 32. This implies that the coding is a compressed representation.
    # We can make lossy compression via this `encoder`.
    encoder = Dense(28^2, args.N, leakyrelu) |> gpu
    decoder = Dense(args.N, 28^2, leakyrelu) |> gpu 

    # Defining main model as a Chain of encoder and decoder models
    m = Chain(encoder, decoder)

    @info("Training model.....")
    loss(x) = mse(m(x), x)
    ## Training
    evalcb = throttle(() -> @show(loss(train_data[1])), args.throttle)
    opt = ADAM(args.lr)
	
    @epochs args.epochs Flux.train!(loss, params(m), zip(train_data), opt, cb = evalcb)
	
    return m, args
end

using Images

img(x::Vector) = Gray.(reshape(clamp.(x, 0, 1), 28, 28))

function sample(m, args)
    imgs = MNIST.images()
    #Converting image of type RGB to float 
    imgs = channelview.(imgs)
    # `args.sample_len` random digits
    before = [imgs[i] for i in rand(1:length(imgs), args.sample_len)]
    # Before and after images
    after = img.(map(x -> cpu(m)(float(vec(x))), before))
    # Stack them all together
    hcat(vcat.(before, after)...)
end

cd(@__DIR__)
m, args= train()
# Sample output
@info("Saving image sample as sample_ae.png")
save("sample_ae.png", sample(m, args))
