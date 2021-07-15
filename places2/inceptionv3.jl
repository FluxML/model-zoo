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

struct InceptionA
    path_1
    path_2
    path_3
    path_4
end

Flux.treelike(InceptionA)

function InceptionA(in_channels, pool_features)
    path_1 = ConvBlock((1, 1), in_channels=>64)

    path_2 = (ConvBlock((1, 1), in_channels=>48),
              ConvBlock((5, 5), 48=>64, pad = (2, 2)))

    path_3 = (ConvBlock((1, 1), in_channels=>64),
              ConvBlock((3, 3), 64=>96, pad = (1, 1)),
              ConvBlock((3, 3), 96=>96, pad = (1, 1)))

    path_4 = (x -> meanpool(x, (3, 3), stride = (1, 1), pad = (1, 1)),
              ConvBlock((1, 1), in_channels=>pool_features))

    InceptionA(path_1, path_2, path_3, path_4)
end

struct InceptionB
    path_1
    path_2
    path_3
end

Flux.treelike(InceptionB)

function InceptionB(in_channels)
    path_1 = ConvBlock((3, 3), in_channels=>384, stride = (2, 2))

    path_2 = (ConvBlock((1, 1), in_channels=>64),
              ConvBlock((3, 3), 64=>96, pad = (1, 1)),
              ConvBlock((3, 3), 96=>96, stride = (2, 2)))

    path_3 = x -> maxpool(x, (3, 3), stride = (2, 2))

    InceptionB(path_1, path_2, path_3)
end

struct InceptionC
    path_1
    path_2
    path_3
    path_4
end

Flux.treelike(InceptionC)

function InceptionC(in_channels, channel_7x7)
    path_1 = ConvBlock((1, 1), in_channels=>192)

    path_2 = (ConvBlock((1, 1), in_channels=>channel_7x7),
              ConvBlock((1, 7), channel_7x7=>channel_7x7, pad = (0, 3)),
              ConvBlock((7, 1), channel_7x7=>192, pad = (3, 0)))

    path_3 = (ConvBlock((1, 1), in_channels=>channel_7x7),
              ConvBlock((7, 1), channel_7x7=>channel_7x7, pad = (3, 0)),
              ConvBlock((1, 7), channel_7x7=>channel_7x7, pad = (0, 3)),
              ConvBlock((7, 1), channel_7x7=>channel_7x7, pad = (3, 0)),
              ConvBlock((1, 7), channel_7x7=>192, pad = (0, 3)))

    path_4 = (x -> meanpool(x, (3, 3), stride = (1, 1), pad = (1, 1)),
              ConvBlock((1, 1), in_channels=>192))

    InceptionC(path_1, path_2, path_3, path_4)
end

struct InceptionD
    path_1
    path_2
    path_3
end

Flux.treelike(InceptionD)

function InceptionD(in_channels)
    path_1 = (ConvBlock((1, 1), in_channels=>192),
              ConvBlock((3, 3), 192=>320, stride = (2, 2)))

    path_2 = (ConvBlock((1, 1), in_channels=>192),
              ConvBlock((1, 7), 192=>192, pad = (0, 3)),
              ConvBlock((7, 1), 192=>192, pad = (3, 0)),
              ConvBlock((3, 3), 192=>192, stride = (2, 2)))

    path_3 = x -> maxpool(x, (3, 3), stride = (2, 2))

    InceptionD(path_1, path_2, path_3)
end

struct InceptionE
    path_1
    path_1_branch
    path_2
    path_2_branch
    path_3
    path_4
end

Flux.treelike(InceptionE)

function InceptionE(in_channels)
    path_1 = ConvBlock((1, 1), in_channels=>384)

    path_1_branch = (ConvBlock((1, 3), 384=>384, pad = (0, 1)),
                     ConvBlock((3, 1), 384=>384, pad = (1, 0)))

    path_2 = (ConvBlock((1, 1), in_channels=>448),
              ConvBlock((3, 3), 448=>384, pad = (1, 1)))

    path_2_branch = (ConvBlock((1, 3), 384=>384, pad = (0, 1)),
                     ConvBlock((3, 1), 384=>384, pad = (1, 0)))

    path_3 = (x -> meanpool(x, (3, 3), stride = (1, 1), pad = (1, 1)),
              ConvBlock((1, 1), in_channels=>192))

    path_4 = ConvBlock((1, 1), in_channels=>320)

    InceptionE(path_1, path_1_branch, path_2, path_2_branch, path_3, path_4)
end

function (c::InceptionE)(x)
    x1 = c.path_1(x)
    x2 = c.path_2[2](c.path_2[1](x))
    cat(3, c.path_1_branch[1](x1), c.path_1_branch[2](x1), c.path_2_branch[1](x2), c.path_2_branch[2](x2), c.path_3[2](c.path_3[1](x)), c.path_4(x))
end

inceptionv3(numclasses = 365) = Chain(
                      ConvBlock((3, 3), 3=>32, stride = (2, 2)),
                      ConvBlock((3, 3), 32=>32),
                      ConvBlock((3, 3), 32=>64, pad = (1, 1)),
                      ConvBlock((1, 1), 64=>80),
                      ConvBlock((3, 3), 80=>192),
                      InceptionA(192, 32),
                      InceptionA(256, 64),
                      InceptionA(288, 64),
                      InceptionB(288),
                      [InceptionC(768, i) for i in [128, 160, 160, 192]]...,
                      InceptionD(768),
                      InceptionE(1280),
                      InceptionE(2048),
                      Dense(2048, numclasses), softmax)

model = inceptionv3() |> gpu

opt = ADAM(params(model))
