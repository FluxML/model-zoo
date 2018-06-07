using Flux
using CuArrays
using Flux: onehotbatch, argmax, @epochs
using Base.Iterators: partition
using BSON: @save, @load

struct ConvBlock
    convlayer
    norm
    nonlinearity
end

Flux.treelike(ConvBlock)

ConvBlock(kernel, chs; stride = (1, 1), pad = (0, 0)) = ConvBlock(Conv(kernel, chs, stride = stride, pad = pad),
                                                                  BatchNorm(chs[2]),
                                                                  x -> relu.(x))

(c::ConvBlock)(x) = c.nonlinearity(c.norm(c.convlayer(x)))

struct Mixed_5b_7a
    branch1
    branch2
    branch3
    branch4
end

Flux.treelike(Mixed_5b_7a)

function Mixed_5b()
    branch1 = ConvBlock((1, 1), 192=>96)

    branch2 = (ConvBlock((1, 1), 192=>48),
               ConvBlock((5, 5), 48=>64, pad = (2, 2)))

    branch3 = (ConvBlock((1, 1), 192=>64),
               ConvBlock((3, 3), 64=>96, pad = (1, 1)),
               ConvBlock((3, 3), 96=>96, pad = (1, 1)))

    branch4 = (x -> meanpool(x, (3, 3), stride = (1, 1), pad = (1, 1)),
               ConvBlock((1, 1), 192=>64, stride = (1, 1)))

    Mixed_5b_7a(branch1, branch2, branch3, branch4)
end

function Mixed_7a()
    branch1 = x -> maxpool(x, (3, 3), stride = (2, 2))

    branch2 = (ConvBlock((1, 1), 1088=>256),
              ConvBlock((3, 3), 256=>384, stride = (2, 2)))

    branch3 = (ConvBlock((1, 1), 1088=>256),
              ConvBlock((3, 3), 256=>288, pad = (1, 1)),
              ConvBlock((3, 3), 288=>320, stride = (2, 2)))

    branch4 = (ConvBlock((1, 1), 1088=>256),
              ConvBlock((3, 3), 256=>288, stride = (2, 2)))

    Mixed_5b_7a(branch1, branch2, branch3, branch4)
end

(m::Mixed_5b)(x) = cat(3, m.branch1(x), m.branch2[2](m.branch2[1](x)), m.branch3[3](m.branch3[2](m.branch3[1](x))), m.branch4[2](m.branch4[1](x)))

struct Block35
    scale
    branch1
    branch2
    branch3
    convlayer
end

Flux.treelike(Block35)

function Block35(scale = 1.0f0)
    branch1 = ConvBlock((1, 1), 320=>32)

    branch2 = (ConvBlock((1, 1), 320=>32),
               ConvBlock((3, 3), 32=>32, pad = (1, 1)))

    branch3 = (ConvBlock((1, 1), 320=>32),
               ConvBlock((3, 3), 32=>48, pad = (1, 1)),
               ConvBlock((3, 3), 48=>64, pad = (1, 1)))

    convlayer = Conv((1, 1), 128=>320)

    Block35(scale, branch1, branch2, branch3, convlayer)
end

(b::Block35)(x) = relu.(b.convlayer(cat(3, b.branch1(x), b.branch2[2](b.branch2[1](x)), b.branch3[3](b.branch3[2](b.branch3[1](x))))) * b.scale + x)

struct Mixed_6a
    branch1
    branch2
    branch3
end

Flux.treelike(Mixed_6a)

function Mixed_6a()
    branch1 = ConvBlock((3, 3), 320=>384, stride = (2, 2))

    branch2 = (ConvBlock((1, 1), 320=>256),
               ConvBlock((3, 3), 256=>256, pad = (1, 1)),
               ConvBlock((3, 3), 256=>384, stride = (2, 2)))

    branch3 = x -> maxpool(x, (3, 3), stride = (2, 2))

    Mixed_6a(branch1, branch2, branch3)
end

(m::Mixed_6a)(x) = cat(3, m.branch1(x), m.branch2[3](m.branch2[2](m.branch2[1](x))), m.branch3(x))

struct Block17
    scale
    branch1
    branch2
    convlayer
end

Flux.treelike(Block17)

function Block17(scale = 1.0f0)
    branch1 = ConvBlock((1, 1), 1088=>192)

    branch2 = (ConvBlock((1, 1), 1088=>128),
               ConvBlock((1, 7), 128=>160, pad = (0, 3)),
               ConvBlock((7, 1), 160=>192, pad = (3, 0)))

    convlayer = Conv((1, 1), 384=>1088)

    Block17(scale, branch1, branch2, convlayer)
end

(b::Block17)(x) = relu.(b.convlayer(cat(3, b.branch1(x), b.branch2[3](b.branch2[2](b.branch2[1](x))))) * b.scale + x)

struct Block8
    scale
    branch1
    branch2
    convlayer
    norelu::Bool
end

Flux.treelike(Block8)

function Block8(scale = 1.0f0, norelu = false)
    branch1 = ConvBlock((1, 1), 2080=>192)

    branch2 = (ConvBlock((1, 1), 2080=>192),
               ConvBlock((1, 3), 192=>224, pad = (0, 1)),
               ConvBlock((3, 1), 224=>256, pad = (1, 0)))

    convlayer = Conv((1, 1), 448=>2080)

    Block8(scale, branch1, branch2, convlayer, norelu)
end

function (b::Block8)(x)
    ẋ = b.convlayer(cat(3, b.branch1(x), b.branch2[3](b.branch2[2](b.branch3[1](x))))) * b.scale + x
    if(!b.norelu)
        ẋ = relu.(ẋ)
    end
    ẋ
end

inceptionresnetv2(nclasses = 365) =
    Chain(ConvBlock((3, 3), 3=>32, stride = (2, 2)),
          ConvBlock((3, 3), 32=>32),
          ConvBlock((3, 3), 32=>64, pad = (1, 1)),
          x -> maxpool(x, (3, 3), stride = (2, 2)),
          ConvBlock((1, 1), 64=>80),
          ConvBlock((3, 3), 80=>192),
          x -> maxpool(x, (3, 3), stride = (2, 2)),
          Mixed_5b(),
          [Block35(0.17f0) for i in 1:10]...,
          Mixed_6a(),
          [Block17(0.10f0) for i in 1:20]...,
          Mixed_7a(),
          [Block8(0.20f0) for i in 1:9]...,
          Block8(1.0f0, true),
          ConvBlock((1, 1), 2080=>1536),
          x -> meanpool(x, (8, 8), stride = (1, 1)),
          Dense(1536, nclasses), softmax)

model = inceptionresnetv2() |> gpu

opt = ADAM(params(model))
