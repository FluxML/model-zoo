using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
using Images
using MLDatasets
using Statistics
using Printf
using Random
using CUDA
CUDA.allowscalar(false)

Base.@kwdef struct HyperParams
    batch_size::Int = 128
    latent_dim::Int = 100
    epochs::Int = 20
    verbose_freq::Int = 1000
    output_x::Int = 6
    output_y::Int = 6
    lr_dscr::Float32 = 0.0002
    lr_gen::Float32 = 0.0002
end

function create_output_image(gen, fixed_noise, hparams)
    fake_images = @. cpu(gen(fixed_noise))
    image_array = reduce(vcat, reduce.(hcat, partition(fake_images, hparams.output_y)))
    image_array = permutedims(dropdims(image_array; dims=(3, 4)), (2, 1))
    image_array = @. Gray(image_array + 1f0) / 2f0
    return image_array
end

  
# weight initialization as given in the paper https://arxiv.org/abs/1511.06434
dcgan_init(shape...) = randn(Float32, shape...) * 0.02f0

function Discriminator()
    return Chain(
            Conv((4, 4), 1 => 64; stride = 2, pad = 1, init = dcgan_init),
            x->leakyrelu.(x, 0.2f0),
            Dropout(0.25),
            Conv((4, 4), 64 => 128; stride = 2, pad = 1, init = dcgan_init),
            x->leakyrelu.(x, 0.2f0),
            Dropout(0.25), 
            x->reshape(x, 7 * 7 * 128, :),
            Dense(7 * 7 * 128, 1))	
end

function Generator(latent_dim::Int)
    return Chain(
            Dense(latent_dim, 7 * 7 * 256),
            BatchNorm(7 * 7 * 256, relu),
            x->reshape(x, 7, 7, 256, :),
            ConvTranspose((5, 5), 256 => 128; stride = 1, pad = 2, init = dcgan_init),
            BatchNorm(128, relu),
            ConvTranspose((4, 4), 128 => 64; stride = 2, pad = 1, init = dcgan_init),
            BatchNorm(64, relu),
            ConvTranspose((4, 4), 64 => 1; stride = 2, pad = 1, init = dcgan_init),
            x -> tanh.(x)
            )
end

# Loss functions
function discriminator_loss(real_output, fake_output)
    real_loss = logitbinarycrossentropy(real_output, 1)
    fake_loss = logitbinarycrossentropy(fake_output, 0)
    return real_loss + fake_loss
end

generator_loss(fake_output) = logitbinarycrossentropy(fake_output, 1)

function train_discriminator!(gen, dscr, x, opt_dscr, hparams)
    noise = randn!(similar(x, (hparams.latent_dim, hparams.batch_size))) 
    fake_input = gen(noise)
    # Taking gradient
    loss, grads = Flux.withgradient(dscr) do dscr
        discriminator_loss(dscr(x), dscr(fake_input))
    end
    update!(opt_dscr, dscr, grads[1])
    return loss
end

function train_generator!(gen, dscr, x, opt_gen, hparams)
    noise = randn!(similar(x, (hparams.latent_dim, hparams.batch_size))) 
    # Taking gradient
    loss, grads = Flux.withgradient(gen) do gen
        generator_loss(dscr(gen(noise)))
    end
    update!(opt_gen, gen, grads[1])
    return loss
end

function train(; kws...)
    # Model Parameters
    hparams = HyperParams(; kws...)

    if CUDA.functional()
        @info "Training on GPU"
    else
        @warn "Training on CPU, this will be very slow!"  # 20 mins/epoch
    end

    # Load MNIST dataset
    images = MLDatasets.MNIST(:train).features
    # Normalize to [-1, 1]
    image_tensor = reshape(@.(2f0 * images - 1f0), 28, 28, 1, :)
    # Partition into batches
    data = [image_tensor[:, :, :, r] |> gpu for r in partition(1:60000, hparams.batch_size)]

    fixed_noise = [randn(Float32, hparams.latent_dim, 1) |> gpu for _=1:hparams.output_x*hparams.output_y]

    # Discriminator
    dscr = Discriminator() |> gpu

    # Generator
    gen =  Generator(hparams.latent_dim) |> gpu

    # Optimizers
    opt_dscr = Flux.setup(Adam(hparams.lr_dscr), dscr)
    opt_gen = Flux.setup(Adam(hparams.lr_gen), gen)

    # Training
    train_steps = 0
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        for x in data
            # Update discriminator and generator
            loss_dscr = train_discriminator!(gen, dscr, x, opt_dscr, hparams)
            loss_gen = train_generator!(gen, dscr, x, opt_gen, hparams)

            if train_steps % hparams.verbose_freq == 0
                @info("Train step $(train_steps), Discriminator loss = $(loss_dscr), Generator loss = $(loss_gen)")
                # Save generated fake image
                output_image = create_output_image(gen, fixed_noise, hparams)
                save(@sprintf("output/dcgan_steps_%06d.png", train_steps), output_image)
            end
            train_steps += 1
        end
    end

    output_image = create_output_image(gen, fixed_noise, hparams)
    save(@sprintf("output/dcgan_steps_%06d.png", train_steps), output_image)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end

