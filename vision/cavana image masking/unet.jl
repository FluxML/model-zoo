#-------Imports-------
using CuArrays, Flux, Images, NNlib, CUDAnative, Statistics
using BSON: @save
using Flux: @treelike, sub2, expand, initn
using Flux.Tracker: track, data, @grad, nobacksies
using Flux.Optimise: @interrupts

#-------Layers not present in Flux-------
function out_size(stride, pad, dilation, kernel, xdims)
    dims = []
    for i in zip(stride, pad, dilation, kernel, xdims)
        push!(dims, i[1] * (i[5] - 1) + (i[4] - 1) * i[3] - 2 * i[2] + 1)
    end
    dims
end

function convtranspose(x, w; stride = 1, pad = 0, dilation = 1)
    stride, pad, dilation = NNlib.padtuple(x, stride), NNlib.padtuple(x, pad), NNlib.padtuple(x, dilation)
    y = similar(x, out_size(stride, pad, dilation, size(w)[1:end-2], size(x)[1:end-2])...,size(w)[end-1],size(x)[end])
    NNlib.∇conv_data(x, y, w, stride = stride, pad = pad, dilation = dilation)
end

convtranspose(x::TrackedArray, w::TrackedArray; kw...) = track(convtranspose, x, w; kw...)
convtranspose(x::AbstractArray, w::TrackedArray; kw...) = track(convtranspose, x, w; kw...)
convtranspose(x::TrackedArray, w::AbstractArray; kw...) = track(convtranspose, x, w; kw...)

@grad convtranspose(x, w; kw...) =
    convtranspose(data.((x, w))...; kw...), Δ -> nobacksies(:convtranspose, (NNlib.conv(data.((Δ, w))...; kw...), NNlib.∇conv_filter(data.((x, Δ, w))...; kw...)))

struct ConvTranspose{N,F,A,V}
    σ::F
    weight::A
    bias::V
    stride::NTuple{N,Int}
    pad::NTuple{N,Int}
    dilation::NTuple{N,Int}
end

ConvTranspose(w::AbstractArray{T,N}, b::AbstractVector{T}, σ = identity;
    stride = 1, pad = 0, dilation = 1) where {T,N} =
    ConvTranspose(σ, w, b, expand.(sub2(Val(N)), (stride, pad, dilation))...)

ConvTranspose(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity; init = Flux.initn,
    stride = 1, pad = 0, dilation = 1) where N =
    ConvTranspose(param(init(k..., ch[2], ch[1])), param(zeros(ch[2])), σ,
                  stride = stride, pad = pad, dilation = dilation)

@treelike ConvTranspose

function (c::ConvTranspose)(x)
    σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
    σ.(convtranspose(x, c.weight, stride = c.stride, pad = c.pad, dilation = c.dilation) .+ b)
end

function Base.show(io::IO, l::ConvTranspose)
    print(io, "ConvTranspose(", size(l.weight)[1:ndims(l.weight)-2])
    print(io, ", ", size(l.weight, ndims(l.weight)-1), "=>", size(l.weight, ndims(l.weight)))
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end

#-------Utilities-------
function center_crop(x, target_size_h, target_size_w)
    start_h = (size(x, 1) - target_size_h) ÷ 2 + 1
    start_w = (size(x, 2) - target_size_w) ÷ 2 + 1
    x[start_h:(start_h + target_size_h - 1), start_w:(start_w + target_size_w - 1), :, :]
end

#-------UNet Architecture-------
UNetConvBlock(in_chs, out_chs, kernel = (3, 3)) =
    Chain(Conv(kernel, in_chs=>out_chs, relu, pad = (1, 1)),
          Conv(kernel, out_chs=>out_chs, relu, pad = (1, 1)))

struct UNetUpBlock
    upsample
    conv_layer
end

@treelike UNetUpBlock

UNetUpBlock(in_chs::Int, out_chs::Int, kernel = (3, 3)) =
    UNetUpBlock(ConvTranspose((2, 2), in_chs=>out_chs, stride=(2, 2)),
                Chain(Conv(kernel, in_chs=>out_chs, relu, pad=(1, 1)),
                      Conv(kernel, out_chs=>out_chs, relu, pad=(1, 1))))

function (u::UNetUpBlock)(x, bridge)
    x = u.upsample(x)
    # Since we know the image dimensions from beforehand we might as well not use the center_crop
    # u.conv_layer(cat(x, center_crop(bridge, size(x, 1), size(x, 2)), dims = 3))
    u.conv_layer(cat(x, bridge, dims = 3))
end

struct UNet
    pool_layer
    conv_blocks
    up_blocks
end

@treelike UNet

# This is to be used for Background and Foreground segmentation
function UNet()
    pool_layer = MaxPool((2, 2))
    conv_blocks = (UNetConvBlock(3, 64), UNetConvBlock(64, 128), UNetConvBlock(128, 256),
                   UNetConvBlock(256, 512), UNetConvBlock(512, 1024))
    up_blocks = (UNetUpBlock(1024, 512), UNetUpBlock(512, 256), UNetUpBlock(256, 128),
                 UNetUpBlock(128, 64), Conv((1, 1), 64=>1))
    UNet(pool_layer, conv_blocks, up_blocks)
end

function (u::UNet)(x)
    outputs = Vector(undef, 5)
    outputs[1] = u.conv_blocks[1](x)
    for i in 2:5
        pool_x = u.pool_layer(outputs[i - 1])
        outputs[i] = u.conv_blocks[i](pool_x)
    end
    up_x = outputs[end]
    for i in 1:4
        up_x = u.up_blocks[i](up_x, outputs[end - i])
    end
    u.up_blocks[end](up_x)
end

#-------Loss and Metric Functions-------
dice_coeff(ŷ, y) = 2 * sum(ŷ .* y) / (sum(ŷ) + sum(y))

function logsig(x)
    max_v = max(zero(x), -x)
    z = CUDAnative.exp(-max_v) + CUDAnative.exp(-x-max_v)
    -(max_v + CUDAnative.log(z))
end

logitbinarycrossentropy(logŷ, y) = (1 - y) * logŷ - logsig(logŷ)

loss(x, y) = mean(logitbinarycrossentropy.(x, y))

#-------Dataset Loading-------
function get_img_paths(data_dir)
    img_path = []
    for i in readdir(data_dir)
        if split(i, ".")[end] == "jpg" || split(i, ".")[end] == "gif"
            push!(img_path, joinpath(data_dir, i))
        end
    end
    img_path
end

loadim(path) = float.(permutedims(channelview(imresize(load(path), 224, 224)), (2, 3, 1)))
loadmask(path) = float.(channelview(imresize(load(path), 224, 224)))[1, :, :]

train_img_paths = get_img_paths("train")
train_mask_paths = get_img_paths("train_masks")
train_imgs = []
train_masks = []
dices = []
losses = []

# Since there are only a few images we can load the entire dataset at once
function load_train_imgs()
    global train_imgs
    imgs_loaded = 0
    for i in train_img_paths
        push!(train_imgs, loadim(i))
        imgs_loaded += 1
        if imgs_loaded % 1000 == 0
            @info "$imgs_loaded Images have been loaded"
        end
    end
end

# Since there are only a few images we can load the entire dataset at once
function load_train_masks()
    global train_masks
    masks_loaded = 0
    for i in train_mask_paths
        push!(train_masks, loadmask(i))
        masks_loaded += 1
        if masks_loaded % 1000 == 0
            @info "$imgs_loaded Masks have been loaded"
        end
    end
end

#-------Training the model-------
function train_model(batch_size = 16, epochs = 10)
    global train_imgs, train_masks, losses, dices
    model = UNet() |> gpu
    running_loss = 0.0
    iters = 0
    opt = ADAM(params(model))

    load_train_imgs()
    load_train_masks()

    train_dataset = [(cat(4, train_imgs[i]...), cat(4, train_masks[i]...)) for i in partition(1:length(train_imgs), batch_size)]

    for epoch in 1:epochs
        for batch in train_dataset
            x = model(batch[1] |> gpu)
            mask = batch[2] |> gpu
            l = loss(x, mask)
            @interrupts Flux.back!(l)
            opt()
            running_loss += l.data
            iters += 1
            if iters % 10 == 0
                push!(losses, cpu(running_loss/iters))
                push!(dices, cpu(dice_coeff(x, mask)))
                @info "Loss after $iters = $(losses[end])"
                @info "Dice Coefficient after $iters = $(dices[end])"
            end
        end
        @info "Epoch $epoch complete. Loss = $(running_loss/iters)"
        @save "unet_model.bson" model
    end
end
