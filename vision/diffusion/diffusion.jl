# Score-Based Generative Modeling
#
# Score-Based Generative Modeling through Stochastic Differential Equations.
# Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, 
# Abhishek Kumar, Stefano Ermon, and Ben Poole.
# https://arxiv.org/pdf/2011.13456.pdf

using MLDatasets
using Flux
using Flux: @functor
using Flux.Data: DataLoader
using Parameters: @with_kw
using BSON
using CUDA
using DrWatson: struct2dict
using Images
using Logging: with_logger
using ProgressMeter: Progress, next!
using TensorBoardLogger: TBLogger, tb_overwrite
using Random

# load MNIST images and return loader
function get_data(batch_size)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    # Normalize to [-1, 1]
    xtrain = reshape(@.(2.0f0 * xtrain - 1.0f0), 28, 28, 1, :)
    DataLoader((xtrain, ytrain), batchsize=batch_size, shuffle=true)
end

"""
Projection of Gaussian Noise onto a time vector.

References:
    paper-  https://arxiv.org/abs/2006.10739
    layers- https://fluxml.ai/Flux.jl/stable/models/basics/#Building-Layers
"""
function GaussianFourierProjection(embed_dim, scale)
    W = randn(Float32, embed_dim ÷ 2) .* scale
    function (t)
        # size(t) => (batch)
        t_proj = t' .* W * 2π
        # returns => (embed_dim, batch)
        return [sin.(t_proj); cos.(t_proj)]
    end
end

"""
Create a UNet architecture as a backbone to a diffusion model.

Notes:
    Images stored in WHCN (width, height, channels, batch) order.
    In our case, MNIST comes in as (28, 28, 1, batch).

References:
    paper-  https://arxiv.org/abs/1505.04597
    Conv-   https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Conv
    TConv-  https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.ConvTranspose
    GNorm-  https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.GroupNorm
"""
struct UNet
    # Embedding
    embed
    # Encoder
    conv1
    dense1
    gnorm1
    conv2
    dense2
    gnorm2
    conv3
    dense3
    gnorm3
    conv4
    dense4
    gnorm4
    # Decoder
    tconv4
    dense5
    tgnorm4
    tconv3
    dense6
    tgnorm3
    tconv2
    dense7
    tgnorm2
    tconv1
    # Scaling Factor
    marginal_prob_std
end

"""
User Facing API for UNet architecture.
"""
function UNet(marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256, scale=30.0f0)
    UNet(
        # Embedding
        Chain(
            GaussianFourierProjection(embed_dim, scale),
            Dense(embed_dim, embed_dim)
        ),
        # Encoding
        Conv((3, 3), 1 => channels[1], stride=1, bias=false),
        Dense(embed_dim, channels[1]),
        GroupNorm(channels[1], 4),
        Conv((3, 3), channels[1] => channels[2], stride=2, bias=false),
        Dense(embed_dim, channels[2]),
        GroupNorm(channels[2], 32),
        Conv((3, 3), channels[2] => channels[3], stride=2, bias=false),
        Dense(embed_dim, channels[3]),
        GroupNorm(channels[3], 32),
        Conv((3, 3), channels[3] => channels[4], stride=2, bias=false),
        Dense(embed_dim, channels[4]),
        GroupNorm(channels[4], 32),
        # Decoding
        ConvTranspose((3, 3), channels[4] => channels[3], stride=2, bias=false),
        Dense(embed_dim, channels[3]),
        GroupNorm(channels[3], 32),

        # FIXME: Julia does not offer a `output_padding` kwarg such as in:
        # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#convtranspose2d
        # 
        # A fix is referenced here: https://github.com/FluxML/Flux.jl/issues/1319
        # and here: https://github.com/FluxML/Flux.jl/issues/829
        # But padding with `SamePad()` or (0, 1, 0, 1) gives (10, 10, 64, 3) which is incorrect
        # 
        # A (incorrect) hack to get the shapes to work is just to pad by (-1, 0, -1, 0):
        # julia> randn(5, 5, 256, 3) |> ConvTranspose((3, 3), 256 => 64, stride=2, pad=(0, -1, 0, -1)) |> size
        # (12, 12, 64, 3)
        #
        # Which is the correct shape, but seems suspicious (negative padding??).
        # Why is passing a negative padding even allowed? 😕
        ConvTranspose((3, 3), channels[3] + channels[3] => channels[2], pad=(0, -1, 0, -1), stride=2, bias=false), Dense(embed_dim, channels[2]),
        GroupNorm(channels[2], 32),
        ConvTranspose((3, 3), channels[2] + channels[2] => channels[1], pad=(0, -1, 0, -1), stride=2, bias=false),
        Dense(embed_dim, channels[1]),
        GroupNorm(channels[1], 32),
        ConvTranspose((3, 3), channels[1] + channels[1] => 1, stride=1, bias=false),
        # Scaling Factor
        marginal_prob_std
    )
end

@functor UNet

expand_dims(x::AbstractVecOrMat, dims::Int=2) = reshape(x, (ntuple(i -> 1, dims)..., size(x)...))

"""
Forward pass of a UNet architecture.
"""
function (unet::UNet)(x, t)
    # Embedding
    embed = swish.(unet.embed(t))
    # Encoder
    h1 = unet.conv1(x)
    h1 .+= expand_dims(unet.dense1(embed))
    h1 = unet.gnorm1(h1)
    h1 = swish.(h1)
    h2 = unet.conv2(h1)
    h2 .+= expand_dims(unet.dense2(embed))
    h2 = unet.gnorm2(h2)
    h2 = swish.(h2)
    h3 = unet.conv3(h2)
    h3 .+= expand_dims(unet.dense3(embed))
    h3 = unet.gnorm3(h3)
    h3 = swish.(h3)
    h4 = unet.conv4(h3)
    h4 .+= expand_dims(unet.dense4(embed))
    h4 = unet.gnorm4(h4)
    h4 = swish.(h4)
    # Decoder
    h = unet.tconv4(h4)
    h .+= expand_dims(unet.dense5(embed))
    h = unet.tgnorm4(h)
    h = swish.(h)
    h = unet.tconv3(cat(h, h3; dims=3))
    h .+= expand_dims(unet.dense6(embed))
    h = unet.tgnorm3(h)
    h = swish.(h)
    h = unet.tconv2(cat(h, h2, dims=3))
    h .+= expand_dims(unet.dense7(embed))
    h = unet.tgnorm2(h)
    h = swish.(h)
    h = unet.tconv1(cat(h, h1, dims=3))
    # Scaling Factor
    h ./ expand_dims(unet.marginal_prob_std(t), 3)
end

function marginal_prob_std(t, sigma=25.0f0)
    sqrt.((sigma .^ (2t) .- 1.0f0) ./ (2.0f0 * log(sigma)))
end

function diffusion_coeff(t, sigma=25.0f0)
    sigma .^ t
end

function model_loss(model, x, device, ϵ=1e-5)
    batch_size = size(x)[end]
    # (batch) of random times to approximate 𝔼[⋅] wrt. t∼𝒰(0, T)
    random_t = rand(Float32, batch_size) .* (1.0f0 - ϵ) .+ ϵ |> device
    # approximate 𝔼[⋅] wrt. x(0)∼𝒫₀(x)
    z = randn(Float32, size(x)) |> device
    std = model.marginal_prob_std(random_t)
    perturbed_x = x + z .* expand_dims(std, 3)
    score = model(perturbed_x, random_t)
    sum(sum((score .* expand_dims(std, 3) + z) .^ 2, dims=(1, 2, 3))) / batch_size
end

function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(x |> cpu, y_size), 28, :)...), (2, 1)))
end

# arguments for the `train` function 
@with_kw mutable struct Args
    η = 1e-4                # learning rate
    batch_size = 32         # batch size
    sample_size = 10        # sampling size for output 
    epochs = 50             # number of epochs
    seed = 0                # random seed
    cuda = false            # use CPU
    verbose_freq = 10       # logging for every verbose_freq iterations
    tblogger = false        # log training with tensorboard
    save_path = "output"    # results path
end

function train(; kws...)
    # load hyperparamters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # GPU config
    if args.cuda && CUDA.has_cuda()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # load MNIST images
    loader = get_data(args.batch_size)

    # initialize UNet model
    unet = UNet(marginal_prob_std) |> device

    # ADAM optimizer
    opt = ADAM(args.η)

    # parameters
    ps = Flux.params(encoder.linear, encoder.μ, encoder.logσ, decoder)

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
            loss, back = Flux.pullback(ps) do
                model_loss(unet, x |> device, device)
            end
            grad = back(1.0f0)
            Flux.Optimise.update!(opt, ps, grad)
            # progress meter
            next!(progress; showvalues=[(:loss, loss)])

            # logging with TensorBoard
            if args.tblogger && train_steps % args.verbose_freq == 0
                with_logger(tblogger) do
                    @info "train" loss = loss
                end
            end

            train_steps += 1
        end
    end

    # save model
    model_path = joinpath(args.save_path, "model.bson")
    let unet = cpu(unet), args = struct2dict(args)
        BSON.@save model_path unet args
        @info "Model saved: $(model_path)"
    end

    # TODO: Add SDE solver to invert from noise
    # _, _, rec_original = reconstuct(encoder, decoder, original, device)
    # rec_original = sigmoid.(rec_original)
    # image = convert_to_image(rec_original, args.sample_size)
    # image_path = joinpath(args.save_path, "epoch_$(epoch).png")
    # save(image_path, image)
    # @info "Image saved: $(image_path)"
end

train()
if abspath(PROGRAM_FILE) == @__FILE__
    train()
end

