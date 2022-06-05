# Score-Based Generative Modeling
#
# Score-Based Generative Modeling through Stochastic Differential Equations.
# Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, 
# Abhishek Kumar, Stefano Ermon, and Ben Poole.
# https://arxiv.org/pdf/2011.13456.pdf

using MLDatasets
using Flux
using Flux: @functor, chunk, params
using Flux.Data: DataLoader
using Parameters: @with_kw
using BSON
using CUDA
using Images
using Logging: with_logger
using ProgressMeter: Progress, next!
using TensorBoardLogger: TBLogger, tb_overwrite
using Random
using Statistics

"""
Projection of Gaussian Noise onto a time vector.

# Notes
This layer will help embed our random times onto the frequency domain. \n
W is not trainable and is sampled once upon construction - see assertions below.

# References
paper-  https://arxiv.org/abs/2006.10739
"""
function GaussianFourierProjection(embed_dim, scale)
    # Instantiate W once
    W = randn(Float32, embed_dim ÷ 2) .* scale
    # Return a function that always references the same W
    function GaussFourierProject(t)
        t_proj = t' .* W * Float32(2π)
        [sin.(t_proj); cos.(t_proj)]
    end
end

"""
Helper function that computes the *standard deviation* of 𝒫₀ₜ(𝘹(𝘵)|𝘹(0)).

# Notes
Derived from the Stochastic Differential Equation (SDE):    \n
                𝘥𝘹 = σᵗ𝘥𝘸,      𝘵 ∈ [0, 1]                   \n

We use properties of SDEs to analytically solve for the stddev
at time t conditioned on the data distribution. \n

We will be using this all over the codebase for computing our model's loss,
scaling our network output, and even sampling new images!
"""
marginal_prob_std(t, sigma=25.0f0) = sqrt.((sigma .^ (2t) .- 1.0f0) ./ 2.0f0 ./ log(sigma))

"""
Create a UNet architecture as a backbone to a diffusion model. \n

# Notes
Images stored in WHCN (width, height, channels, batch) order. \n
In our case, MNIST comes in as (28, 28, 1, batch). \n

# References
paper-  https://arxiv.org/abs/1505.04597
"""
struct UNet
    layers::NamedTuple
end

"""
User Facing API for UNet architecture.
"""
function UNet(channels=[32, 64, 128, 256], embed_dim=256, scale=30.0f0)
    return UNet((
        gaussfourierproj=GaussianFourierProjection(embed_dim, scale),
        linear=Dense(embed_dim, embed_dim, swish),
        # Encoding
        conv1=Conv((3, 3), 1 => channels[1], stride=1, bias=false),
        dense1=Dense(embed_dim, channels[1]),
        gnorm1=GroupNorm(channels[1], 4, swish),
        conv2=Conv((3, 3), channels[1] => channels[2], stride=2, bias=false),
        dense2=Dense(embed_dim, channels[2]),
        gnorm2=GroupNorm(channels[2], 32, swish),
        conv3=Conv((3, 3), channels[2] => channels[3], stride=2, bias=false),
        dense3=Dense(embed_dim, channels[3]),
        gnorm3=GroupNorm(channels[3], 32, swish),
        conv4=Conv((3, 3), channels[3] => channels[4], stride=2, bias=false),
        dense4=Dense(embed_dim, channels[4]),
        gnorm4=GroupNorm(channels[4], 32, swish),
        # Decoding
        tconv4=ConvTranspose((3, 3), channels[4] => channels[3], stride=2, bias=false),
        dense5=Dense(embed_dim, channels[3]),
        tgnorm4=GroupNorm(channels[3], 32, swish),
        tconv3=ConvTranspose((3, 3), channels[3] + channels[3] => channels[2], pad=(0, -1, 0, -1), stride=2, bias=false),
        dense6=Dense(embed_dim, channels[2]),
        tgnorm3=GroupNorm(channels[2], 32, swish),
        tconv2=ConvTranspose((3, 3), channels[2] + channels[2] => channels[1], pad=(0, -1, 0, -1), stride=2, bias=false),
        dense7=Dense(embed_dim, channels[1]),
        tgnorm2=GroupNorm(channels[1], 32, swish),
        tconv1=ConvTranspose((3, 3), channels[1] + channels[1] => 1, stride=1, bias=false),
    ))
end

@functor UNet

"""
Helper function that adds `dims` dimensions to the front of a `AbstractVecOrMat`.
Similar in spirit to TensorFlow's `expand_dims` function.

# References:
https://www.tensorflow.org/api_docs/python/tf/expand_dims
"""
expand_dims(x::AbstractVecOrMat, dims::Int=2) = reshape(x, (ntuple(i -> 1, dims)..., size(x)...))

"""
Makes the UNet struct callable and shows an example of a "Functional" API for modeling in Flux. \n
"""
function (unet::UNet)(x, t)
    # Embedding
    embed = unet.layers.gaussfourierproj(t)
    embed = unet.layers.linear(embed)
    # Encoder
    h1 = unet.layers.conv1(x)
    h1 = h1 .+ expand_dims(unet.layers.dense1(embed), 2)
    h1 = unet.layers.gnorm1(h1)
    h2 = unet.layers.conv2(h1)
    h2 = h2 .+ expand_dims(unet.layers.dense2(embed), 2)
    h2 = unet.layers.gnorm2(h2)
    h3 = unet.layers.conv3(h2)
    h3 = h3 .+ expand_dims(unet.layers.dense3(embed), 2)
    h3 = unet.layers.gnorm3(h3)
    h4 = unet.layers.conv4(h3)
    h4 = h4 .+ expand_dims(unet.layers.dense4(embed), 2)
    h4 = unet.layers.gnorm4(h4)
    # Decoder
    h = unet.layers.tconv4(h4)
    h = h .+ expand_dims(unet.layers.dense5(embed), 2)
    h = unet.layers.tgnorm4(h)
    h = unet.layers.tconv3(cat(h, h3; dims=3))
    h = h .+ expand_dims(unet.layers.dense6(embed), 2)
    h = unet.layers.tgnorm3(h)
    h = unet.layers.tconv2(cat(h, h2, dims=3))
    h = h .+ expand_dims(unet.layers.dense7(embed), 2)
    h = unet.layers.tgnorm2(h)
    h = unet.layers.tconv1(cat(h, h1, dims=3))
    # Scaling Factor
    h ./ expand_dims(marginal_prob_std(t), 3)
end

"""
Model loss following the denoising score matching objectives:

# Notes
Denoising score matching objective:
```julia
min wrt. θ (
    𝔼 wrt. 𝘵 ∼ 𝒰(0, 𝘛)[
        λ(𝘵) * 𝔼 wrt. 𝘹(0) ∼ 𝒫₀(𝘹) [
            𝔼 wrt. 𝘹(t) ∼ 𝒫₀ₜ(𝘹(𝘵)|𝘹(0)) [
                (||𝘚₀(𝘹(𝘵), 𝘵) - ∇ log [𝒫₀ₜ(𝘹(𝘵) | 𝘹(0))] ||₂)²
            ]
        ]
    ]
)
``` 
Where 𝒫₀ₜ(𝘹(𝘵) | 𝘹(0)) and λ(𝘵), are available analytically and
𝘚₀(𝘹(𝘵), 𝘵) is estimated by a U-Net architecture.

# References:
http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf \n
https://yang-song.github.io/blog/2021/score/#estimating-the-reverse-sde-with-score-based-models-and-score-matching \n
https://yang-song.github.io/blog/2019/ssm/
"""
function model_loss(model, x, ϵ=1.0f-5)
    batch_size = size(x)[end]
    # (batch) of random times to approximate 𝔼[⋅] wrt. 𝘪 ∼ 𝒰(0, 𝘛)
    random_t = rand!(similar(x, batch_size)) .* (1.0f0 - ϵ) .+ ϵ
    # (batch) of perturbations to approximate 𝔼[⋅] wrt. 𝘹(0) ∼ 𝒫₀(𝘹)
    z = randn!(similar(x))
    std = expand_dims(marginal_prob_std(random_t), 3)
    # (batch) of perturbed 𝘹(𝘵)'s to approximate 𝔼 wrt. 𝘹(t) ∼ 𝒫₀ₜ(𝘹(𝘵)|𝘹(0))
    perturbed_x = x + z .* std
    # 𝘚₀(𝘹(𝘵), 𝘵)
    score = model(perturbed_x, random_t)
    # mean over batches
    mean(
        # L₂ norm over WHC dimensions
        sum((score .* std + z) .^ 2; dims=1:(ndims(x) - 1))
    )
end

"""
Helper function that loads MNIST images and returns loader.
"""
function get_data(batch_size)
    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtrain = reshape(xtrain, 28, 28, 1, :)
    DataLoader((xtrain, ytrain), batchsize=batch_size, shuffle=true)
end

"""
Helper function from DrWatson.jl to convert a struct to a dict
"""
function struct2dict(::Type{DT}, s) where {DT<:AbstractDict}
    DT(x => getfield(s, x) for x in fieldnames(typeof(s)))
end
struct2dict(s) = struct2dict(Dict, s)

# arguments for the `train` function 
@with_kw mutable struct Args
    η = 1e-4                                        # learning rate
    batch_size = 32                                 # batch size
    epochs = 50                                     # number of epochs
    seed = 1                                        # random seed
    cuda = false                                    # use CPU
    verbose_freq = 10                               # logging for every verbose_freq iterations
    tblogger = true                                 # log training with tensorboard
    save_path = "output"                            # results path
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
    unet = UNet() |> device

    # ADAM optimizer
    opt = ADAM(args.η)

    # parameters
    ps = Flux.params(unet)

    !ispath(args.save_path) && mkpath(args.save_path)

    # logging by TensorBoard.jl
    if args.tblogger
        tblogger = TBLogger(args.save_path, tb_overwrite)
    end

    # Training
    train_steps = 0
    @info "Start Training, total $(args.epochs) epochs"
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader))

        for (x, _) in loader
            x = device(x)
            loss, grad = Flux.withgradient(ps) do
                model_loss(unet, x)
            end
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
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end