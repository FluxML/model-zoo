include("../layers/resnetblock.jl")
include("../layers/separableconv.jl")

struct ResnetSeparableConv
    conv_layers
    norm_layers
    pooling_layer
    shortcut
end

Flux.treelike(ResnetSeparableConv)

function (block::ResnetSeparableConv)(input)
    local value = copy.(input)
    for i in 1:length(block.conv_layers)
        value = relu.((block.norm_layers[i])((block.conv_layers[i])(value)))
    end
    block.pooling_layer(value) + block.shortcut(input)
end

function ResnetSeparableConv(filters, kernels::Array{Tuple{Int,Int}}, pads::Array{Tuple{Int,Int}}, strides::Array{Tuple{Int,Int}}, pooling, shortcut = identity)
    local conv_layers = []
    local norm_layers = []
    for i in 2:length(filters)
        push!(conv_layers, SeparableConv(kernels[i-1], filters[i-1]=>filters[i], pad = pads[i-1], stride = strides[i-1]))
        push!(norm_layers, BatchNorm(filters[i]))
    end
    ResnetSeparableConv(Tuple(conv_layers), Tuple(norm_layers), pooling , shortcut)
end

function ResnetSeparableConv(filters, kernels::Array{Int}, pads::Array{Int}, strides::Array{Int}, pooling, shortcut = identity)
    ResnetSeparableConv(filters, [(i,i) for i in kernels], [(i,i) for i in pads], [(i,i) for i in strides], pooling, shortcut)
end

function ResnetSeparableConv_1(filters, chs = filters[1]=>filters[2])
    ResnetSeparableConv(filters, [3,3], [1,1], [1,1], x -> maxpool(x, (3,3), stride=(2,2), pad=(1,1)), Conv((1,1), chs, stride=(2,2)))
end

ResnetSeparableConv_2() = ResnetSeparableConv([728,728,728,728], [3,3,3], [1,1,1], [1,1,1], identity)

function build_model()
    local top = [Conv((3,3), 3=>32, stride=(2,2), pad=(1,1)),
        BatchNorm(32),
        x -> relu.(x),
        Conv((3,3), 32=>64, pad=(1,1)),
        BatchNorm(64),
        x -> relu.(x),
        ResnetSeparableConv_1([64,128,128]),
        ResnetSeparableConv_1([128,256,256]),
        ResnetSeparableConv_1([256,728,728])]
    local mid = [ResnetSeparableConv_2() for i in 1:8]
    local bottom = [ResnetSeparableConv_1([728,728,1024], 728=>728),
        SeparableConv((3,3), 1024=>1536, pad=(1,1)),
        BatchNorm(1536),
        x -> relu.(x),
        SeparableConv((3,3), 1536=>2048, pad=(1,1)),
        BatchNorm(2048),
        x -> relu.(x),
        x -> maxpool(x, (size(x,1),size(x,2))),
        Dense(2048, 1000),
        softmax]
    Chain(top..., mid..., bottom...)
end
