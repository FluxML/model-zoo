using Flux
using Base.Iterators: partition
using CuArrays
using Flux: @epochs
using BSON: @save, @load

struct InceptionBlock
    path_1
    path_2
    path_3
    path_4
end

ConvBlock(kernel, chs; stride = (1, 1), pad = (0, 0)) = Chain(Conv(kernel, chs, stride = stride, pad = pad),
							      BatchNorm(chs[2])
							      x -> relu.(x))

function InceptionBlock(in_chs, chs_1x1, chs_3x3_reduce, chs_3x3, chs_5x5_reduce, chs_5x5, pool_proj)
    path_1 = ConvBlock((1, 1), in_chs=>chs_1x1)
    path_2 = Chain(ConvBlock((1, 1), in_chs=>chs_3x3_reduce),
		   ConvBlock((3, 3), chs_3x3_reduce=>chs_3x3), pad = (1, 1))
    path_3 = Chain(ConvBlock((1, 1), in_chs=>chs_5x5_reduce),
		   ConvBlock((5, 5), chs_5x5_reduce=>chs_5x5, pad = (1,1)))
    path_4 = Chain(x -> maxpool(x, (3,3), stride = (1, 1), pad = (1, 1)),
		   ConvBlock((1, 1), in_chs=>pool_proj))
    InceptionBlock(path_1, path_2, path_3, path_4)
end

(m::InceptionBlock)(x) = cat(3, m.path_1(x), m.path_2(x), m.path_3(x), m.path_4(x))

googlenet(num_classes = 365) =
    Chain(ConvBlock((7, 7), 3=>64, stride = (2, 2), pad = (3, 3)),
      x -> maxpool(x, (3, 3), stride = (2, 2), pad = (1, 1)),
      ConvBlock((1, 1), 64=>64),
      ConvBlock((3, 3), 64=>192, pad = (1, 1)),
      x -> maxpool(x, (3, 3), stride = (2, 2), pad = (1, 1)),
      InceptionBlock(192, 64, 96, 128, 16, 32, 32),
      InceptionBlock(256, 128, 128, 192, 32, 96, 64),
      x -> maxpool(x, (3, 3), stride = (2, 2), pad = (1, 1)),
      InceptionBlock(480, 192, 96, 208, 16, 48, 64),
      InceptionBlock(512, 160, 112, 224, 24, 64, 64),
      InceptionBlock(512, 128, 128, 256, 24, 64, 64),
      InceptionBlock(512, 112, 144, 288, 32, 64, 64),
      InceptionBlock(528, 112, 144, 288, 32, 128, 128)
      x -> maxpool(x, (3, 3), stride = (2, 2), pad = (1, 1)),
      InceptionBlock(832, 256, 160, 320, 32, 128, 128),
      InceptionBlock(832, 384, 192, 384, 48, 128, 128),
      x -> meanpool(x, (7, 7), stride = (1, 1), pad = (0, 0)),
      x -> reshape(x, :, size(x, 4)),
      Dense(1024, num_classes), softmax)

model = googlenet() |> gpu

opt = ADAM(params(model))

info("Model exported to GPU")

model(rand(224, 224, 3, 10) |> gpu)

info("Model works on random data")
