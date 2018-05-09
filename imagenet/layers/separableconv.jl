using Flux
using Flux: initn
using NNlib: conv

struct DepthwiseConv
    σ
    weight
    bias
    stride
    pad
end

Flux.treelike(DepthwiseConv)

function DepthwiseConv(k::NTuple{N,Integer}, in_chs::Int, depth_mul::Int, σ = identity; init = initn, stride::NTuple{N,Integer} = map(_->1,k), pad::NTuple{N,Integer} = map(_->0,k)) where N
    weights = [param(init(k...,1,depth_mul)) for i in 1:in_chs]
    bias = [param(zeros(depth_mul)) for i in 1:in_chs]
    DepthwiseConv(σ, Tuple(weights), Tuple(bias), stride, pad)
end

function (c::DepthwiseConv)(x)
    b = reshape(c.bias[1], map(_->1, c.stride)..., :, 1)
    x2 = σ.(conv(x[:,:,1:1,:], c.weight[1], stride = c.stride, pad = c.pad) .+ b)
    for i in 2:length(c.bias)
        b = reshape(c.bias[i], map(_->1, c.stride)..., :, 1)
        x2 = cat(3, x2, σ.(conv(x[:,:,i:i,:], c.weight[i], stride = c.stride, pad = c.pad) .+ b))
    end
    x2
end

PointwiseConv(chs, σ = relu) = Conv((1,1), chs, σ)

struct SeparableConv
    depth
    point
end

Flux.treelike(SeperableConv)

function SeparableConv(k::NTuple{N,Integer}, chs::Pair{<:Integer,<:Integer}, depth_mul::Int = 1; stride::NTuple{N,Integer} = map(_->1,k), pad::NTuple{N,Integer} = map(_->0,k)) where N
    SeperableConv(DepthwiseConv(k, chs[1], depth_mul, stride = stride, pad = pad), PointwiseConv(chs[1]*depth_mul=>chs[2]))
end

function (c::SeparableConv)(x)
    c.point(c.depth(x))
end
