include("residualblock.jl")

# Function to generate the residual blocks for ResNet18 and ResNet34
function BasicBlock(filters::Int, downsample::Bool = false, res_top::Bool = false)
    if(!downsample || res_top)
        return ResidualBlock([filters for i in 1:3], [3,3], [1,1], [1,1])
    end
    shortcut = Chain(Conv((3,3), filters÷2=>filters, pad = (1,1), stride = (2,2)), BatchNorm(filters))
    ResidualBlock([filters÷2, filters, filters], [3,3], [1,1], [1,2], shortcut)
end

# Function to generate the residual blocks used for ResNet50, ResNet101 and ResNet152
function Bottleneck(filters::Int, downsample::Bool = false, res_top::Bool = false)
    if(!downsample && !res_top)
        return ResidualBlock([4 * filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,1])
    elseif(downsample && res_top)
        return ResidualBlock([filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,1], Chain(Conv((1,1), filters=>4 * filters, pad = (0,0), stride = (1,1)), BatchNorm(4 * filters)))
    else
        shortcut = Chain(Conv((1,1), 2 * filters=>4 * filters, pad = (0,0), stride = (2,2)), BatchNorm(4 * filters))
        return ResidualBlock([2 * filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,2], shortcut)
    end
end

struct StandardResnet
    top
    residual
    bottom
end

Flux.treelike(StandardResnet)

# Function to build Standard Resnet models as described in the paper "Deep Residual Learning for Image Recognition"
function StandardResnet(Block, layers, initial_filters::Int = 64, nclasses::Int = 1000)

    local top = []
    local residual = []
    local bottom = []

    push!(top, Conv((7,7), 3=>initial_filters, pad = (3,3), stride = (2,2)))
    push!(top, x -> maxpool(x, (3,3), pad = (1,1), stride = (2,2)))

    for i in 1:length(layers)
        push!(residual, Block(initial_filters, true, i==1))
        for j in 2:layers[i]
            push!(residual, Block(initial_filters))
        end
        initial_filters *= 2
    end

    push!(bottom, x -> meanpool(x, (7,7)))
    push!(bottom, x -> reshape(x, :, size(x,4)))
    if(Block == Bottleneck)
        push!(bottom, (Dense(2048, nclasses)))
    else
        push!(bottom, (Dense(512, nclasses)))
    end
    push!(bottom, softmax)

    StandardResnet(Chain(top...), Chain(residual...), Chain(bottom...))
end

function (model::StandardResnet)(input)
    model.bottom(model.residual(model.top(input)))
end
