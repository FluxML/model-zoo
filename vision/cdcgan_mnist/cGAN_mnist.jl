using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
using Images
using ImageMagick
using MLDatasets
using Statistics
using Parameters: @with_kw
using Random
using Printf
using CUDA
using Zygote

if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

@with_kw struct HyperParams
    batch_size::Int = 128
    latent_dim::Int = 100
    nclasses::Int = 10
    epochs::Int = 25
    verbose_freq::Int = 1000
    output_x::Int = 6        # No. of sample images to concatenate along x-axis 
    output_y::Int = 6        # No. of sample images to concatenate along y-axis
    lr_dscr::Float64 = 0.0002
    lr_gen::Float64 = 0.0002
end

struct Discriminator
    d_labels		# Submodel to take labels as input and convert them to the shape of image ie. (28, 28, 1, batch_size)
    d_common   
end

function discriminator(args)
    d_labels = Chain(Dense(args.nclasses,784), x-> reshape(x, 28, 28, 1, size(x, 2))) |> gpu
    d_common = Chain(Conv((3,3), 2=>128, pad=(1,1), stride=(2,2)),
                  x-> leakyrelu.(x, 0.2f0),
                  Dropout(0.4),
                  Conv((3,3), 128=>128, pad=(1,1), stride=(2,2), leakyrelu),
                  x-> leakyrelu.(x, 0.2f0),
                  x-> reshape(x, :, size(x, 4)),
                  Dropout(0.4),
                  Dense(6272, 1)) |> gpu
    Discriminator(d_labels, d_common)
end

function (m::Discriminator)(x, y)
    t = cat(m.d_labels(x), y, dims=3)
    return m.d_common(t)
end

struct Generator
    g_labels          # Submodel to take labels as input and convert it to the shape of (7, 7, 1, batch_size) 
    g_latent          # Submodel to take latent_dims as input and convert it to shape of (7, 7, 128, batch_size)
    g_common    
end

function generator(args)
    g_labels = Chain(Dense(args.nclasses, 49), x-> reshape(x, 7 , 7 , 1 , size(x, 2))) |> gpu
    g_latent = Chain(Dense(args.latent_dim, 6272), x-> leakyrelu.(x, 0.2f0), x-> reshape(x, 7, 7, 128, size(x, 2))) |> gpu
    g_common = Chain(ConvTranspose((4, 4), 129=>128; stride=2, pad=1),
            BatchNorm(128, leakyrelu),
            Dropout(0.25),
            ConvTranspose((4, 4), 128=>64; stride=2, pad=1),
            BatchNorm(64, leakyrelu),
            Conv((7, 7), 64=>1, tanh; stride=1, pad=3)) |> gpu
    Generator(g_labels, g_latent, g_common)
end

function (m::Generator)(x, y)
    t = cat(m.g_labels(x), m.g_latent(y), dims=3)
    return m.g_common(t)
end

function load_data(hparams)
    # MLDatasets.MNIST.download(i_accept_the_terms_of_use=true)

    # Load MNIST dataset
    images, labels = MNIST(:train)[:]
    # Normalize to [-1, 1]
    image_tensor = reshape(@.(2f0 * images - 1f0), 28, 28, 1, :)
    y = float.(Flux.onehotbatch(labels, 0:hparams.nclasses-1))
    # Partition into batches
    data = [(image_tensor[:, :, :, r], y[:, r]) |> gpu for r in partition(1:60000, hparams.batch_size)]
    return data
end

# Loss functions
function discr_loss(real_output, fake_output)
    real_loss = logitbinarycrossentropy(real_output, 1f0)
    fake_loss = logitbinarycrossentropy(fake_output, 0f0)
    return (real_loss + fake_loss)
end

generator_loss(fake_output) = logitbinarycrossentropy(fake_output, 1f0)

function train_discr(discr, fake_data, fake_labels, original_data, label, opt_discr)
    ps = params(discr.d_labels, discr.d_common)
    loss, back = Zygote.pullback(ps) do
            discr_loss(discr(label, original_data), discr(fake_labels, fake_data))
    end
    grads = back(1f0)
    update!(opt_discr, ps, grads)
    return loss
end

Zygote.@nograd train_discr

function train_gan(gen, discr, original_data, label, opt_gen, opt_discr, hparams)
    # Random Gaussian Noise and Labels as input for the generator
    noise = randn!(similar(original_data, (hparams.latent_dim, hparams.batch_size)))
    labels = rand(0:hparams.nclasses-1, hparams.batch_size)
    y = Flux.onehotbatch(labels, 0:hparams.nclasses-1)
    noise , y  = noise, float.(y) |> gpu

    ps = params(gen.g_labels, gen.g_latent, gen.g_common)
    loss = Dict()
    loss["gen"], back = Zygote.pullback(ps) do
            fake = gen(y, noise)
            loss["discr"] = train_discr(discr, fake, y, original_data, label, opt_discr)
            generator_loss(discr(y, fake))
    end
    grads = back(1f0)
    update!(opt_gen, ps, grads)
    return loss
end

function create_output_image(gen, fixed_noise, fixed_labels, hparams)
    @eval Flux.istraining() = false
    fake_images = @. cpu(gen(fixed_labels, fixed_noise))
    @eval Flux.istraining() = true
    image_array = permutedims(dropdims(reduce(vcat, reduce.(hcat, partition(fake_images, hparams.output_y))); dims=(3, 4)), (2, 1))
    image_array = @. Gray(image_array + 1f0) / 2f0
    return image_array
end

function train(; kws...)
    hparams = HyperParams(kws...)

    # Load the data
    data = load_data(hparams)

    fixed_noise = [randn(hparams.latent_dim, 1) |> gpu for _=1:hparams.output_x * hparams.output_y]

    fixed_labels = [float.(Flux.onehotbatch(rand(0:hparams.nclasses-1, 1), 0:hparams.nclasses-1)) |> gpu 
                             for _ =1:hparams.output_x * hparams.output_y]

    # Discriminator
    dscr = discriminator(hparams) 

    # Generator
    gen =  generator(hparams)

    # Optimizers
    opt_dscr = ADAM(hparams.lr_dscr, (0.5, 0.99))
    opt_gen = ADAM(hparams.lr_gen, (0.5, 0.99))

    # Check if the `output` directory exists or needed to be created
    isdir("output")||mkdir("output")

    # Training
    train_steps = 0
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        for (x, y) in data
            # Update discriminator and generator
            loss = train_gan(gen, dscr, x, y, opt_gen, opt_dscr, hparams)

            if train_steps % hparams.verbose_freq == 0
                @info("Train step $(train_steps), Discriminator loss = $(loss["discr"]), Generator loss = $(loss["gen"])")
                # Save generated fake image
                output_image = create_output_image(gen, fixed_noise, fixed_labels, hparams)
                save(@sprintf("output/cgan_steps_%06d.png", train_steps), output_image)
            end

            train_steps += 1
        end
    end

    output_image = create_output_image(gen, fixed_noise, fixed_labels, hparams)
    save(@sprintf("output/cgan_steps_%06d.png", train_steps), output_image)
    return Flux.onecold.(cpu(fixed_labels))
end    

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end

        
