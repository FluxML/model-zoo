# weight initialization
function random_normal(shape...)
    return map(Float32,rand(Normal(0,0.02),shape...))
end

ConvBlock(in_ch::Int,out_ch::Int,k=4,s=2,p=1) = 
    Chain(Conv((3,3), in_ch=>out_ch,pad = (p, p), stride=(1,1);init=random_normal),
          BatchNormWrap(out_ch)...,
          x->leakyrelu.(x,0.2),
	  Conv((k,k), out_ch=>out_ch,pad = (p, p), stride=(s,s);init=random_normal),
	  BatchNormWrap(out_ch)...,
          x->leakyrelu.(x,0.2))

function Discriminator()
    model = Chain(Conv((4,4), 6=>64,pad = (1, 1), stride=(2,2);init=random_normal),BatchNormWrap(64)...,x->leakyrelu.(x,0.2),
                  ConvBlock(64,128),
                  ConvBlock(128,256),
                  ConvBlock(256,512,4,1,1),
                  Conv((4,4), 512=>1,pad = (1, 1), stride=(1,1);init=random_normal))
                  x->σ.(x))
    return model 
end
