# weight initialization
function _random_normal(shape...)
    return map(Float32,rand(Normal(0,0.02),shape...))
end

_Conv(in_chs::Int,out_chs::Int,k=3,s=1,p=1) = 
				Chain(Conv((k,k),in_chs=>out_chs,pad=(p,p),stride=(s,s);init=_random_normal),
				x -> leakyrelu.(x,0.2))

_ConvBN(in_chs::Int,out_chs::Int,k=3,s=1,p=1) = 
				Chain(Conv((k,k),in_chs=>out_chs,pad=(p,p),stride=(s,s);init=_random_normal),
				BatchNormWrap(out_chs)...,
				x -> leakyrelu.(x,0.2))

function print_(x)
	println("SIze of x : $(size(x))")
end

function Discriminator()
	Chain(_Conv(3,64,3,1,1),
		  _ConvBN(64,64,3,2,1),
		  _ConvBN(64,128,3,1,1),
		  _ConvBN(128,128,3,2,1),
		  _ConvBN(128,256,3,1,1),
		  _ConvBN(256,256,3,2,1),
		  _ConvBN(256,512,3,1,1),
		  _ConvBN(512,512,3,2,1),
		  # x -> print_(x),
		  x -> flatten(x),
		  # x -> print_(x),
		  Dense(126 * 86 * 512,1024),
		  x -> leakyrelu.(x,0.2),
		  Dense(1024,1),
		  x -> σ.(x))
end
