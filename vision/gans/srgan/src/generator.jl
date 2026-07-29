# weight initialization
function random_normal(shape...)
    return map(Float32,rand(Normal(0,0.02),shape...))
end

_Conv(in_chs::Int,out_chs::Int,k=3,s=1,p=1) = Chain(Conv((k,k),in_chs=>out_chs,pad=(p,p),stride=(s,s);init=random_normal))
_ConvBN(in_chs::Int,out_chs::Int) = Chain(Conv((3,3),in_chs=>out_chs,pad=(1,1),stride=(1,1);init=random_normal),
										 BatchNormWrap(out_chs)...)

ConvBlock() = Chain(Conv((3,3),64=>64,pad=(1,1),stride=(1,1);init=random_normal),
                    BatchNormWrap(64)...,
					PRelu(64))

mutable struct ResidualBlock
	conv_blocks
end

@treelike ResidualBlock

ResidualBlock() = ResidualBlock((ConvBlock(),ConvBlock()))

function (m::ResidualBlock)(x)
	out = m.conv_blocks[1](x)
	out = m.conv_blocks[2](out)
	out .+ x
end

UpBlock(in_chs::Int,out_chs::Int) = Chain(_Conv(in_chs,out_chs,3,1,1),
										  x->PixelShuffle(x,UP_SAMPLE_FACTOR_STEP),
										  PRelu(div(out_chs,UP_SAMPLE_FACTOR_STEP * UP_SAMPLE_FACTOR_STEP)))
mutable struct Generator
	init_conv
	residual_blocks
	conv_blocks
	up_blocks
end

@treelike Generator

function Gen(B::Int)
	println("B : $B")
	init_conv = Chain(_Conv(3,64,9,1,1),PRelu(64))

	residual_blocks = []
	for i in 1:B
		push!(residual_blocks,ResidualBlock())
	end

	residual_blocks = tuple(residual_blocks...)

	conv_blocks = (_ConvBN(64,64),_Conv(64,3,9,1,1))

	up_blocks = (UpBlock(64,256),UpBlock(64,256))

	Generator(init_conv,residual_blocks,conv_blocks,up_blocks)
end

function (gen::Generator)(x)
	println("Input size : $(size(x))")
	x = gen.init_conv(x)
	first_conv_x = x

	println("FirstConv size : $(size(x))")
	for res_block in gen.residual_blocks
		x = res_block(x)
	end

	println("ResBlock size : $(size(x))")
	x = gen.conv_blocks[1](x)
	x = x .+ first_conv_x

	println("ResConvBlock size : $(size(x))")
	for up_block in gen.up_blocks
		println("UP!")
		x = up_block(x)
	end

	println("Done 1")
	x = gen.conv_blocks[2](x)
	println("Done 2")

	tanh.(x)
end
