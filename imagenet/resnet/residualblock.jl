using Flux

struct ResidualBlock
    conv_layers
    norm_layers
    shortcut
end

Flux.treelike(ResidualBlock)

# ResidualBlock Function allows us to define a Residual Block having any number of Convolution and Batch Normalization Layers
function ResidualBlock(filters, kernels::Array{Tuple{Int,Int}}, pads::Array{Tuple{Int,Int}}, strides::Array{Tuple{Int,Int}}, shortcut = identity)
    local conv_layers = []
    local norm_layers = []
    for i in 2:length(filters)
        push!(conv_layers, Conv(kernels[i-1], filters[i-1]=>filters[i], pad = pads[i-1], stride = strides[i-1]))
        push!(norm_layers, BatchNorm(filters[i]))
    end
    ResidualBlock(Tuple(conv_layers),Tuple(norm_layers),shortcut)
end

# Function converts the Array of scalar kernel, pad and stride values to tuples
function ResidualBlock(filters, kernels::Array{Int}, pads::Array{Int}, strides::Array{Int}, shortcut = identity)
    ResidualBlock(filters, [(i,i) for i in kernels], [(i,i) for i in pads], [(i,i) for i in strides], shortcut)
end

function (block::ResidualBlock)(input)
    local value = copy.(input)
    for i in 1:length(block.conv_layers)-1
        value = relu.((block.norm_layers[i])((block.conv_layers[i])(value)))
    end
    relu.(((block.norm_layers[end])((block.conv_layers[end])(value))) + block.shortcut(input))
end
