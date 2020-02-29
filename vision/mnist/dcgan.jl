using Base.Iterators: partition
using Flux
using Flux.Data.MNIST
using Flux.Optimise: update!
using Flux: logitbinarycrossentropy
using Images
using Statistics
using Printf

const BATCH_SIZE = 128
const NOISE_DIM = 100
const EPOCHS = 20
const VERBOSE_FREQ = 1000
const OUTPUT_X = 6
const OUTPUT_Y = 6
const LR_DSCR = 0.0002
const LR_GEN = 0.0002

# Taking vector of images and return minibatch
function make_minibatch(xs)
    ys = reshape(Float32.(reduce(hcat, xs)), 28, 28, 1, :)
    return @. 2f0 * ys - 1f0
end

function create_output_image(gen, fixed_noise)
    @eval Flux.istraining() = false
    fake_images = @. cpu(gen(fixed_noise))
    @eval Flux.istraining() = true
    image_array = dropdims(reduce(vcat, reduce.(hcat, partition(fake_images, OUTPUT_Y))); dims=(3, 4))
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

function train_discriminator!(gen, dscr, batch, opt_dscr)
    noise = randn(Float32, NOISE_DIM, BATCH_SIZE) |> gpu
    fake_input = gen(noise)
    ps = Flux.params(dscr)
    # Taking gradient
    loss, back = Flux.pullback(ps) do
        discriminator_loss(dscr(batch), dscr(fake_input))
    end
    grad = back(1f0)
    update!(opt_dscr, ps, grad)
    return loss
end

function train_generator!(gen, dscr, batch, opt_gen)
    noise = randn(Float32, NOISE_DIM, BATCH_SIZE) |> gpu
    ps = Flux.params(gen)
    # Taking gradient
    loss, back = Flux.pullback(ps) do
        generator_loss(dscr(gen(noise)))
    end
    grad = back(1f0)
    update!(opt_gen, ps, grad)
    return loss
end

function train()
    # MNIST dataset
    images = MNIST.images()
    data = [make_minibatch(xs) |> gpu for xs in partition(images, BATCH_SIZE)]

    fixed_noise = [randn(NOISE_DIM, 1) |> gpu for _=1:OUTPUT_X*OUTPUT_Y]

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
        Dense(NOISE_DIM, 7 * 7 * 256),
        BatchNorm(7 * 7 * 256, relu),
        x->reshape(x, 7, 7, 256, :),
        ConvTranspose((5, 5), 256 => 128; stride = 1, pad = 2),
        BatchNorm(128, relu),
        ConvTranspose((4, 4), 128 => 64; stride = 2, pad = 1),
        BatchNorm(64, relu),
        ConvTranspose((4, 4), 64 => 1, tanh; stride = 2, pad = 1),
        ) |> gpu

    # Optimizers
    opt_dscr = ADAM(LR_DSCR)
    opt_gen = ADAM(LR_GEN)

    # Training
    train_steps = 0
    for ep in 1:EPOCHS
        @info "Epoch $ep"
        for batch in data
            # Update discriminator and generator
            loss_dscr = train_discriminator!(gen, dscr, batch, opt_dscr)
            loss_gen = train_generator!(gen, dscr, batch, opt_gen)

            if train_steps % VERBOSE_FREQ == 0
                @info("Train step $(train_steps), Discriminator loss = $(loss_dscr), Generator loss = $(loss_gen)")
                # Save generated fake image
                output_image = create_output_image(gen, fixed_noise)
                save(@sprintf("dcgan_steps_%06d.png", train_steps), output_image)
            end
            train_steps += 1
        end
    end

    output_image = create_output_image(gen, fixed_noise)
    save(@sprintf("dcgan_steps_%06d.png", train_steps), output_image)
end

cd(@__DIR__)
train()