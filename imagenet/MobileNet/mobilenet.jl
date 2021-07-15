include("../layers/separableconv.jl")

ConvLayer1(kernel, chs; stride=(1,1), pad=(0,0)) = Chain(Conv(kernel, chs, relu, stride=stride, pad=pad), BatchNorm(chs[2]))

ConvLayer2(kernel, in_chs, stride=(1,1), pad=(1,1)) = Chain(DepthwiseConv(kernel, in_chs, 1, relu, stride=stride, pad=pad), BatchNorm(in_chs))

m = Chain(ConvLayer1((3,3), 3=>32, stride=(2,2), pad=(1,1)),
        ConvLayer2((3,3), 32),
        ConvLayer1((1,1), 32=>64),
        ConvLayer2((3,3), 64, stride=(2,2)),
        ConvLayer1((1,1), 64=>128),
        ConvLayer2((3,3), 128),
        ConvLayer1((1,1), 128=>128),
        ConvLayer2((3,3), 128, stride=(2,2)),
        ConvLayer1((1,1), 128=>256),
        ConvLayer2((3,3), 256),
        ConvLayer1((1,1), 256=>256),
        ConvLayer2((3,3), 256, stride=(2,2)),
        ConvLayer1((1,1), 256=>512),
        [i%2==1? ConvLayer2((3,3), 512) : ConvLayer1((1,1), 512) for i in 1:10]...,
        ConvLayer2((3,3), 512, stride=(2,2)),
        ConvLayer1((1,1), 512=>1024),
        ConvLayer2((3,3), 1024, stride=(2,2), pad=(4,4)),
        ConvLayer1((1,1), 1024=>1024),
        x -> meanpool(x, (7,7), stride=(1,1), pad=(0,0)),
        Dense(1024, 1000), softmax)
