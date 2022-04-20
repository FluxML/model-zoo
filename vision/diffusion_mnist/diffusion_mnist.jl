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
using DrWatson: struct2dict
using Images
using Logging: with_logger
using ProgressMeter: Progress, next!
using TensorBoardLogger: TBLogger, tb_overwrite
using Random

"""
Projection of Gaussian Noise onto a time vector.

# Notes
This layer will help embed our random times onto the frequency domain. \n
W is not trainable and is sampled once upon construction - see assertions below.

# References
paper-  https://arxiv.org/abs/2006.10739 \n
layers- https://fluxml.ai/Flux.jl/stable/models/basics/#Building-Layers
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
function marginal_prob_std(t, sigma=25.0f0)
    sqrt.((sigma .^ (2t) .- 1.0f0) ./ 2.0f0 ./ log(sigma))
end

"""
Create a UNet architecture as a backbone to a diffusion model. \n

# Notes
Images stored in WHCN (width, height, channels, batch) order. \n
In our case, MNIST comes in as (28, 28, 1, batch). \n

# References
paper-  https://arxiv.org/abs/1505.04597 \n
Conv-   https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Conv \n
TConv-  https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.ConvTranspose \n
GNorm-  https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.GroupNorm \n
Flux-   https://fluxml.ai/Flux.jl/stable/models/advanced/#Custom-Model-Example
"""
struct UNet
    # ------------------- ------------------- ------------------- --------------
    # Embedding
    # ------------------- ------------------- ------------------- --------------
    gaussfourierproj
    linear::Dense
    # ------------------- ------------------- ------------------- --------------
    # Encoder
    # ------------------- ------------------- ------------------- --------------
    conv1::Conv
    dense1::Dense
    gnorm1::GroupNorm
    # ------------------- ------------------- -------------------
    conv2::Conv
    dense2::Dense
    gnorm2::GroupNorm
    # ------------------- -------------------
    conv3::Conv
    dense3::Dense
    gnorm3::GroupNorm
    # -------------------
    conv4::Conv
    dense4::Dense
    gnorm4::GroupNorm
    # -------
    # Decoder
    # -------
    tconv4::ConvTranspose
    dense5::Dense
    tgnorm4::GroupNorm
    # ------------------- 
    tconv3::ConvTranspose
    dense6::Dense
    tgnorm3::GroupNorm
    # ------------------- -------------------
    tconv2::ConvTranspose
    dense7::Dense
    tgnorm2::GroupNorm
    # ------------------- ------------------- -------------------
    tconv1::ConvTranspose
    # ------------------- ------------------- ------------------- --------------
    # Scaling Factor
    # ------------------- ------------------- ------------------- --------------
    marginal_prob_std
    # ------------------- ------------------- ------------------- --------------
end

"""
User Facing API for UNet architecture.
"""
function UNet(marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256, scale=30.0f0)
    UNet(
        # Embedding
        GaussianFourierProjection(embed_dim, scale),
        Dense(embed_dim, embed_dim),
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
        ########################################################################
        # FIXME: Julia does not offer a `output_padding` kwarg such as in:
        # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#convtranspose2d
        #
        # A fix is referenced here: https://github.com/FluxML/Flux.jl/issues/1319
        # and here: https://github.com/FluxML/Flux.jl/issues/829
        # But padding with `pad=SamePad()` or `pad=(0, 1, 0, 1)` yields (10, 10, 64, 3) 
        # which is still incorrect.
        # 
        # Based off relationship 14 of https://arxiv.org/pdf/1603.07285.pdf:
        # 𝘰' = 𝘴(𝘪' - 1) + 𝘢 + 𝘬 - 2𝘱
        # Set 𝘰' = 12, 𝘴 = 2, 𝘪' = 5, 𝘬 = 3, 𝘱 = 0, then
        # 12 = 2(5 - 1) + a + 3 - 2(0) = 11 + a => a = 1
        # However, no Flux API that I could find exposes the argument a for the user...
        #
        # A (incorrect) hack to get the shapes to work is just to pad by (-1, 0, -1, 0):
        # julia> randn(5, 5, 256, 3) |> ConvTranspose((3, 3), 256 => 64, stride=2, pad=(0, -1, 0, -1)) |> size
        # (12, 12, 64, 3)
        #
        # Which is the correct shape, but seems suspicious (negative padding??).
        # Why is passing a negative padding even allowed in the first place? 😕
        ########################################################################
        ConvTranspose((3, 3), channels[3] + channels[3] => channels[2], pad=(0, -1, 0, -1), stride=2, bias=false),
        Dense(embed_dim, channels[2]),
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

"""
Helper function that adds `dims` dimensions to the front of a `AbstractVecOrMat`.
Similar in spirit to TensorFlow's `expand_dims` function.

# References:
https://www.tensorflow.org/api_docs/python/tf/expand_dims
"""
expand_dims(x::AbstractVecOrMat, dims::Int=2) = reshape(x, (ntuple(i -> 1, dims)..., size(x)...))

"""
Helper function that reverses the order of dimensions.
"""
reverse_dims(x) = permutedims(x, reverse(ntuple(i -> i, length(size(x)))))

"""
Makes the UNet struct callable and shows an example of a "Functional" API for modeling in Flux. \n

Notes:
    ```julia
    @assert size(a) == (28, 28, 1, 32, 32)
    @assert size(b) == (32, 32)
    o = reverse_dims(a) .+ reverse_dims(b)' |> reverse_dims
    size(o)  # (28, 28, 1, 32, 32)
    ```
    Is used to broadcast an operation from b to a 
    without the cost of a copy (using expand_dims for example)
    which is not allowed (mutating vector) by Zygote in 
    a custom model.
"""
function (unet::UNet)(x, t)
    # Embedding
    embed = unet.gaussfourierproj(t)
    embed = swish.(unet.linear(embed))
    # Encoder
    h1 = unet.conv1(x)
    h1 = reverse_dims(h1) .+ unet.dense1(embed)' |> reverse_dims
    h1 = unet.gnorm1(h1)
    h1 = swish.(h1)
    h2 = unet.conv2(h1)
    h2 = reverse_dims(h2) .+ unet.dense2(embed)' |> reverse_dims
    h2 = unet.gnorm2(h2)
    h2 = swish.(h2)
    h3 = unet.conv3(h2)
    h3 = reverse_dims(h3) .+ unet.dense3(embed)' |> reverse_dims
    h3 = unet.gnorm3(h3)
    h3 = swish.(h3)
    h4 = unet.conv4(h3)
    h4 = reverse_dims(h4) .+ unet.dense4(embed)' |> reverse_dims
    h4 = unet.gnorm4(h4)
    h4 = swish.(h4)
    # Decoder
    h = unet.tconv4(h4)
    h = reverse_dims(h) .+ unet.dense5(embed)' |> reverse_dims
    h = unet.tgnorm4(h)
    h = swish.(h)
    h = unet.tconv3(cat(h, h3; dims=3))
    h = reverse_dims(h) .+ unet.dense6(embed)' |> reverse_dims
    h = unet.tgnorm3(h)
    h = swish.(h)
    h = unet.tconv2(cat(h, h2, dims=3))
    h = reverse_dims(h) .+ unet.dense7(embed)' |> reverse_dims
    h = unet.tgnorm2(h)
    h = swish.(h)
    h = unet.tconv1(cat(h, h1, dims=3))
    # Scaling Factor
    reverse_dims(h) ./ unet.marginal_prob_std(t) |> reverse_dims
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
function model_loss(model, x, device, ϵ=1.0f-5)
    batch_size = size(x)[end]
    # (batch) of random times to approximate 𝔼[⋅] wrt. 𝘪 ∼ 𝒰(0, 𝘛)
    random_t = rand(Float32, batch_size) .* (1.0f0 - ϵ) .+ ϵ |> device
    # (batch) of perturbations to approximate 𝔼[⋅] wrt. 𝘹(0) ∼ 𝒫₀(𝘹)
    z = randn(Float32, size(x)) |> device
    std = expand_dims(model.marginal_prob_std(random_t), 3)
    # (batch) of perturbed 𝘹(𝘵)'s to approximate 𝔼 wrt. 𝘹(t) ∼ 𝒫₀ₜ(𝘹(𝘵)|𝘹(0))
    perturbed_x = x + z .* std
    # 𝘚₀(𝘹(𝘵), 𝘵)
    score = model(perturbed_x, random_t)
    # mean over batches
    sum(
        # L₂ norm over WHC dimensions
        sum(
            (score .* std + z) .^ 2,
            dims=(1, 2, 3)
        )
    ) / batch_size
end

"""
Helper function that loads MNIST images and returns loader.
"""
function get_data(batch_size)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtrain = reshape(xtrain, 28, 28, 1, :)
    DataLoader((xtrain, ytrain), batchsize=batch_size, shuffle=true)
end

# arguments for the `train` function 
@with_kw mutable struct Args
    η = 1e-4                                        # learning rate
    batch_size = 32                                 # batch size
    epochs = 30                                     # number of epochs
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
    unet = UNet(marginal_prob_std) |> device

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
            loss, grad = Flux.withgradient(ps) do
                model_loss(unet, x |> device, device)
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