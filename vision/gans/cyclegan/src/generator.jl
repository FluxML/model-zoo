# weight initialization
function _random_normal(shape...)
    return map(Float32,rand(Normal(0,0.02),shape...))
end

UNetConvBlock(in_chs, out_chs, kernel = (3, 3)) =
    Chain(Conv(kernel, in_chs=>out_chs,pad = (1, 1);init=_random_normal) ,BatchNormWrap(out_chs)...,x->leakyrelu.(x,0.2),
          Conv(kernel, out_chs=>out_chs,pad = (1, 1);init=_random_normal),BatchNormWrap(out_chs)...,x->leakyrelu.(x,0.2))

ConvDown(in_chs,out_chs,kernel = (4,4)) = Chain(Conv(kernel,out_chs=>out_chs,pad=(1,1),stride=(2,2);init=_random_normal),
                                                BatchNormWrap(out_chs)...,
                                                x->leakyrelu.(x,0.2)) # Convolution And Downsample

struct UNetUpBlock
    upsample
    conv_layer
end

@treelike UNetUpBlock

UNetUpBlock(in_chs::Int, out_chs::Int, kernel = (3, 3)) =
    UNetUpBlock(ConvTranspose((2, 2), in_chs=>out_chs, stride=(2, 2);init=_random_normal),
                Chain(Conv(kernel, in_chs=>out_chs,pad = (1, 1);init=_random_normal),BatchNormWrap(out_chs)...,x->leakyrelu.(x,0.2),
                Conv(kernel, out_chs=>out_chs,pad = (1, 1);init=_random_normal),BatchNormWrap(out_chs)...,x->leakyrelu.(x,0.2)))

function (u::UNetUpBlock)(x, bridge)
    x = u.upsample(x)
    u.conv_layer(cat(x, bridge, dims = 3))
end

struct UNet
    conv_down_blocks
    conv_blocks
    up_blocks
end

@treelike UNet

function UNet()
    conv_down_blocks = (ConvDown(64,64),ConvDown(128,128),ConvDown(256,256),ConvDown(512,512))
    conv_blocks = (UNetConvBlock(3, 64), UNetConvBlock(64, 128), UNetConvBlock(128, 256),
                   UNetConvBlock(256, 512), UNetConvBlock(512, 1024))
    up_blocks = (UNetUpBlock(1024, 512), UNetUpBlock(512, 256), UNetUpBlock(256, 128),
                 UNetUpBlock(128, 64), Conv((1, 1), 64=>3;init=_random_normal))
    UNet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::UNet)(x)
    outputs = Vector(undef, 5)
    outputs[1] = u.conv_blocks[1](x)
    for i in 2:5
        pool_x = u.conv_down_blocks[i - 1](outputs[i - 1])
        outputs[i] = u.conv_blocks[i](pool_x)
    end
    up_x = outputs[end]
    for i in 1:4
        up_x = u.up_blocks[i](up_x, outputs[end - i])
    end
    tanh.(u.up_blocks[end](up_x))
end
