# Variational Autoencoder(VAE)
#
# Auto-Encoding Variational Bayes
# Diederik P Kingma, Max Welling
# https://arxiv.org/abs/1312.6114

using JLD2
# using CUDA, cuDNN  # uncomment one of these if you use GPU
# using AMDGPU
# using Metal
using DrWatson: struct2dict
using Flux
using Optimisers: AdamW
using MLUtils: randn_like, chunk, DataLoader
using Flux: logitbinarycrossentropy
using Images
using Logging: with_logger
using MLDatasets
using ProgressMeter: Progress, next!
using TensorBoardLogger: TBLogger, tb_overwrite
using Random

# load MNIST images and return loader
function get_data(batch_size)
    xtrain, ytrain = MLDatasets.MNIST(split=:train)[:]
    xtrain = reshape(xtrain, 28^2, :)
    return DataLoader((xtrain, ytrain), batchsize=batch_size, shuffle=true)
end

struct Encoder
    linear
    μ
    logσ
end

Flux.@layer Encoder

Encoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Encoder(
    Dense(input_dim, hidden_dim, tanh),   # linear
    Dense(hidden_dim, latent_dim),        # μ
    Dense(hidden_dim, latent_dim),        # logσ
)

function (encoder::Encoder)(x)
    h = encoder.linear(x)
    encoder.μ(h), encoder.logσ(h)
end

Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Chain(
    Dense(latent_dim, hidden_dim, tanh),
    Dense(hidden_dim, input_dim)
)

function reconstuct(encoder, decoder, x)
    μ, logσ = encoder(x)
    z = μ + randn_like(logσ) .* exp.(logσ)
    return μ, logσ, decoder(z)
end

function model_loss(encoder, decoder, x)
    μ, logσ, decoder_z = reconstuct(encoder, decoder, x)
    batch_size = size(x)[end]
    # KL-divergence
    kl_q_p = 0.5f0 * sum(@. (exp(2*logσ) + μ^2 - 1 - 2*logσ)) / batch_size

    logp_x_z = -logitbinarycrossentropy(decoder_z, x, agg=sum) / batch_size
    
    return -logp_x_z + kl_q_p
end

function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(x |> cpu, y_size), 28, :)...), (2, 1)))
end

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 1e-3                # learning rate
    λ = 1e-4                # regularization paramater
    batch_size = 128        # batch size
    sample_size = 10        # sampling size for output    
    epochs = 20             # number of epochs
    seed = 0                # random seed
    use_gpu = true              # use GPU
    input_dim = 28^2        # image size
    latent_dim = 64          # latent dimension
    hidden_dim = 500        # hidden dimension
    verbose_freq = 10       # logging for every verbose_freq iterations
    tblogger = false        # log training with tensorboard
    save_path = "output"    # results path
end

function train(; kws...)
    # load hyperparamters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    if args.use_gpu
        device = Flux.get_device()
    else
        device = Flux.get_device("CPU")
    end

    @info "Training on $device"

    # load MNIST images
    loader = get_data(args.batch_size)
    
    # initialize encoder and decoder
    encoder = Encoder(args.input_dim, args.latent_dim, args.hidden_dim) |> device
    decoder = Decoder(args.input_dim, args.latent_dim, args.hidden_dim) |> device

    # ADAM optimizer
    opt_enc = Flux.setup(AdamW(eta=args.η, lambda=args.λ), encoder)
    opt_dec = Flux.setup(AdamW(eta=args.η, lambda=args.λ), decoder)

    !ispath(args.save_path) && mkpath(args.save_path)

    # logging by TensorBoard.jl
    if args.tblogger
        tblogger = TBLogger(args.save_path, tb_overwrite)
    end

    # fixed input
    original, _ = first(get_data(args.sample_size^2))
    original = original |> device
    image = convert_to_image(original, args.sample_size)
    image_path = joinpath(args.save_path, "original.png")
    save(image_path, image)

    # training
    train_steps = 0
    @info "Start Training, total $(args.epochs) epochs"
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader))

        for (x, _) in loader 
            x_dev = x |> device
            loss, (grad_enc, grad_dec) = Flux.withgradient(encoder, decoder) do enc, dec
                model_loss(enc, dec, x_dev)
            end
    
            Flux.update!(opt_enc, encoder, grad_enc)
            Flux.update!(opt_dec, decoder, grad_dec)
            # progress meter
            next!(progress; showvalues=[(:loss, loss)]) 

            # logging with TensorBoard
            if args.tblogger && train_steps % args.verbose_freq == 0
                with_logger(tblogger) do
                    @info "train" loss=loss
                end
            end

            train_steps += 1
        end
        # save image
        _, _, rec_original = reconstuct(encoder, decoder, original)
        rec_original = sigmoid.(rec_original)
        image = convert_to_image(rec_original, args.sample_size)
        image_path = joinpath(args.save_path, "epoch_$(epoch).png")
        save(image_path, image)
        @info "Image saved: $(image_path)"
    end

    # save model
    let encoder = cpu(encoder), decoder = cpu(decoder), args=struct2dict(args)
        filepath = joinpath(args[:save_path], "checkpoint.jld2") 
        JLD2.save(filepath, "encoder", Flux.state(encoder),
                            "decoder", Flux.state(decoder),
                            "args", args)                            
        @info "Model saved: $(filepath)"
    end
end

if abspath(PROGRAM_FILE) == @__FILE__ 
    train()
end

