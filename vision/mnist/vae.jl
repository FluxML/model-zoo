using Flux, Flux.Data.MNIST, Statistics
using Flux: throttle, params
using Images
using Parameters: @with_kw
using Juno: @progress
# Extend distributions slightly to have a numerically stable logpdf for `p` close to 1 or 0.
using Distributions
import Distributions: logpdf

logpdf(b::Bernoulli, y::Bool) = y * log(b.p + eps(Float32)) + (1f0 - y) * log(1 - b.p + eps(Float32))

@with_kw mutable struct Args
    lr::Float64 = 3e-3    # learning rate
    epochs::Int = 20    # Number of epochs for training model
    Dz::Int = 5    # latent dimensionality
    Dh::Int = 500    # Hidden units
    throttle::Int = 30
    batchsize::Int = 100 
end

function get_processed_data(args)
    # Load data, binarise it, and partition into mini-batches of M.
    X = float.(hcat(vec.(MNIST.images())...)) .> 0.5
    N = size(X, 2)
    data = [X[:,i] for i in Iterators.partition(1:N, args.batchsize)]
    return data, X
end

# Function for generative model
Generative_model(args) = Chain(Dense(args.Dz, args.Dh, tanh), Dense(args.Dh, 28^2, σ))

# logp(x|z) - conditional probability of data given latents.
logp_x_z(x, z, f) = sum(logpdf.(Bernoulli.(f(z)), x))

# KL-divergence between approximation posterior and N(0, 1) prior.
kl_q_p(μ, logσ) = 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1f0 .- (2 .* logσ))

function train(; kws...)
    args = Args(; kws...)
	
    # Loading Data
    data, X = get_processed_data(args)
    
    @info("Constructing Encoder-Decoder models....")
    # Components of recognition model / "encoder" MLP.
    A, μ, logσ = Dense(28^2, args.Dh, tanh), Dense(args.Dh, args.Dz), Dense(args.Dh, args.Dz)
    g(X) = (h = A(X); (μ(h), logσ(h)))
    z(μ, logσ) = μ + exp(logσ) * randn(Float32)

    # Generative model / "decoder" MLP.
    f = Generative_model(args)

    ## Define ways of doing things with the model
	
    # Monte Carlo estimator of mean ELBO using M samples.
    L̄(X) = ((μ̂, logσ̂) = g(X); (logp_x_z(X, z.(μ̂, logσ̂), f) - kl_q_p(μ̂, logσ̂)) * 1 // args.batchsize)
	
    loss(X) = -L̄(X) + 0.01f0 * sum(x->sum(x.^2), params(f))

    ## Training
    evalcb = throttle(() -> @show(-L̄(X[:, rand(1:size(X, 2), args.batchsize)])), args.throttle)	# Callback function
    opt = ADAM(args.lr)    # ADAM optimizer
    ps = params(A, μ, logσ, f)    # Defining parmeters to be trained
	
    @info("Beginning Training.....")
      @progress for i = 1:args.epochs
        @info "Epoch $i"
        Flux.train!(loss, ps, zip(data), opt, cb=evalcb)
    end
	
    return f, z, args
end

# Sample from the learned model.
modelsample(f, z, args) = rand.(Bernoulli.(f(z.(zeros(args.Dz), zeros(args.Dz)))))

# Converting Sampled data into Gray image of size `(28,28)`
img(x) = Gray.(reshape(x, 28, 28))

cd(@__DIR__)
f, z, args = train()
sample = hcat(img.([modelsample(f, z, args) for i = 1:10])...)
@info("Saving Image as `sample.png`.....")
save("sample.png", sample)
