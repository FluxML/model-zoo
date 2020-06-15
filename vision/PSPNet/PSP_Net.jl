using Flux
using NNlib
#= Here we make residual blocks like that in resnet architecture.number_of_classes should be equal to the number of labels that you want to segment=#
function Residual_Block(input,Filters::Tuple,number_of_classes)#Filters will be a tuple deciding the number of kernels in each layer.

    conv1=Conv((1,1),number_of_classes =>Filters[1], pad = 0, dilation = 1)
    batch1=BatchNorm(Filters[1])
    conv2=Conv((3,3),Filters[1]=>Filters[2], pad = 1, dilation = 1)
    batch2=BatchNorm(Filters[2])
    conv3=Conv((1,1),Filters[2] => Filters[3], pad = 0, dilation = 1)
    batch3=BatchNorm(Filters[3])

    conv_skip=Conv((3,3), number_of_classes=>Filters[3];
        stride = 1, pad = 1, dilation = 1)
    batch_skip=BatchNorm(Filters[3])

    X_output=Chain(conv1,batch1,x->leakyrelu.(x,0.01),conv2,batch2,x->leakyrelu.(x,0.01),conv3,batch3,x->leakyrelu.(x,0.01))
    X_Skip_output=Chain(conv_skip,batch_skip)
    sum=X_Skip_output(input)+X_output(input)
    non_linearity=Chain(x->leakyrelu.(x,0.01))
    return non_linearity(sum)

end


function feature_maps(input,number_of_classes
    # base covolution module to get input image feature maps
    O1 = Residual_Block(input,(32,32,64),number_of_classes)
    O2 = Residual_Block(O1,(64,64,128),size(O1)[3])
    O3 = Residual_Block(O2,(128,128,256),size(O2)[3])
    return O3
end
x=ones(256,256,3,1)

function upsample(x,ratio)
  (h, w, c, n) = size(x)
  y = similar(x, (1, ratio[1], 1, ratio[2], 1, 1))
  fill!(y, 1)
  z = reshape(x, (h, 1, w, 1, c, n))  .* y
  reshape(permutedims(z, (2,1,4,3,5,6)), size(x) .* ratio)
end

function pyramid_feature_maps(input,number_of_classes)
    base=feature_maps(input,number_of_classes)

    mp1=MeanPool((1,1); pad = 0, stride = 1)
    ct1=Conv((1,1), 256 =>64 ,stride = 1, pad = 0, dilation = 1)

    mp2=MeanPool((2,2); pad = 0, stride = 2)
    ct2=ConvTranspose((1,1),  256=> 64, stride = 1, pad = 0, dilation = 1)

    mp3=MeanPool((4,4); pad = 0, stride = 4)
    ct3=ConvTranspose((1,1),  256=>64 ,stride = 1, pad = 0, dilation = 1)

    mp4=MeanPool((8,8); pad = 0, stride = 8)
    ct4=ConvTranspose((1,1),  256=>64, stride = 1, pad = 0, dilation = 1)


    Pyramid_feature1=Chain(mp1,ct1,x->upsample(x,(1,1,1,1)))
    Pyramid_feature2=Chain(mp2,ct2,x->upsample(x,(2,2,1,1)))
    Pyramid_feature3=Chain(mp3,ct3,x->upsample(x,(4,4,1,1)))
    Pyramid_feature4=Chain(mp4,ct4,x->upsample(x,(8,8,1,1)))
    matrices=(base,Pyramid_feature1(base),Pyramid_feature2(base),Pyramid_feature3(base),Pyramid_feature4(base))
    concat_matrices=cat(matrices...;dims=(3))
    return concat_matrices
end

function last_conv_module(input,number_of_classes)
    X = pyramid_feature_maps(input,number_of_classes)
    conv_last=Conv((3,3),size(X)[3]=>number_of_classes, pad = 1, dilation = 1)
    batch_last=BatchNorm(number_of_classes)
    last_layer=Chain(conv_last,batch_last,x->flatten(x))
    return last_layer(X)
end
