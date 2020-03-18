using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux: logitbinarycrossentropy
using Images
using MLDatasets
using Statistics
using Parameters: @with_kw
using Printf
using Random
using Zygote: @nograd

@with_kw struct HyperParams
    batch_size::Int = 128
    latent_dim::Int = 100
    epochs::Int = 20
    verbose_freq::Int = 1000
    output_x::Int = 6
    output_y::Int = 6
    lr_dscr::Float64 = 0.0002
    lr_gen::Float64 = 0.0002
end

function create_output_image(gen, fixed_noise, hparams)
    @eval Flux.istraining() = false
    fake_images = @. cpu(gen(fixed_noise))
    @eval Flux.istraining() = true
    image_array = dropdims(reduce(vcat, reduce.(hcat, partition(fake_images, hparams.output_y))); dims=(3, 4))
    image_array = @. Gray(image_array + 1f0) / 2f0
    return image_array
end

# Loss functions
function discriminator_loss(real_output, fake_output)
    real_loss = mean(logitbinarycrossentropy.(real_output, 1f0))
    fake_loss = mean(logitbinarycrossentropy.(fake_output, 0f0))
    return real_loss + fake_loss
end

generator_loss(fake_output) = mean(logitbinarycrossentropy.(fake_output, 1f0))

function train_discriminator!(dscr, fake, x, opt_dscr, hparams)
    ps = Flux.params(dscr)
    # Taking gradient
    loss, back = Flux.pullback(ps) do
        discriminator_loss(dscr(x), dscr(fake))
    end
    grad = back(1f0)
    update!(opt_dscr, ps, grad)
    return loss
end

@nograd train_discriminator!

function train_gen_dscr!(gen, dscr, x, opt_gen, opt_dscr, hparams)
    noise = randn!(similar(x, (hparams.latent_dim, hparams.batch_size))) 
    ps_gen = Flux.params(gen)
    # Taking gradient

    loss_dscr = nothing
    loss_gen, back_gen = Flux.pullback(ps_gen) do
        fake = gen(noise)
        loss_dscr = train_discriminator!(dscr, fake, x, opt_dscr, hparams)
        generator_loss(dscr(fake))
    end

    grad_gen = back_gen(1f0)
    update!(opt_gen, ps_gen, grad_gen)

    return loss_gen, loss_dscr
end

function train(; kws...)
    # Model Parameters
    hparams = HyperParams(; kws...)

    # Load MNIST dataset
    images, _ = MLDatasets.MNIST.traindata(Float32)
    # Normalize to [-1, 1] and convert it to WHCN
    image_tensor = permutedims(reshape(@.(2f0 * images - 1f0), 28, 28, 1, :), (2, 1, 3, 4))
    # Partition into batches
    data = [image_tensor[:, :, :, r] |> gpu for r in partition(1:60000, hparams.batch_size)]

    fixed_noise = [randn(hparams.latent_dim, 1) |> gpu for _=1:hparams.output_x*hparams.output_y]

    # Discriminator
    dscr =  Chain(
        Conv((4, 4), 1 => 64; stride = 2, pad = 1),
        x->leakyrelu.(x, 0.2f0),
        Dropout(0.25),
        Conv((4, 4), 64 => 128; stride = 2, pad = 1),
        x->leakyrelu.(x, 0.2f0),
        Dropout(0.25), 
        x->reshape(x, 7 * 7 * 128, :),
        Dense(7 * 7 * 128, 1)) |> gpu

    # Generator
    gen = Chain(
        Dense(hparams.latent_dim, 7 * 7 * 256),
        BatchNorm(7 * 7 * 256, relu),
        x->reshape(x, 7, 7, 256, :),
        ConvTranspose((5, 5), 256 => 128; stride = 1, pad = 2),
        BatchNorm(128, relu),
        ConvTranspose((4, 4), 128 => 64; stride = 2, pad = 1),
        BatchNorm(64, relu),
        ConvTranspose((4, 4), 64 => 1, tanh; stride = 2, pad = 1),
        ) |> gpu

    # Optimizers
    opt_dscr = ADAM(hparams.lr_dscr)
    opt_gen = ADAM(hparams.lr_gen)

    # Training
    train_steps = 0
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        for x in data
            # Update discriminator and generator
            loss_gen, loss_dscr = train_gen_dscr!(gen, dscr, x, opt_gen, opt_dscr, hparams)

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

cd(@__DIR__)
train()
