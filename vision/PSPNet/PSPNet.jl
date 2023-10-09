#Semantic Segmentation on CityScape dataset
#with the convolutional neural network know as PSPNET.
#This script also combines various
# packages from the Julia ecosystem  with Flux.

using Flux
using NNlib
using Parameters: @with_kw
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using CuArrays
import BSON: @save, @load
using CUDAapi
using ProgressMeter
using FileIO
using Images
import DrWatson: savename, struct2dict
using Statistics, Random

#We create our model which has three parts
@with_kw mutable struct PSPNet
    #first part is making the residual blocks,In total there are three residual blocks where each block contains three convolutional layers in series and one skip connection
    number_of_classes = 3
    Filters = (32,32,64)
    #Start of first residual block
    conv1 = Conv((1,1),number_of_classes=>Filters[1], pad = 0, dilation = 1)
    batch1 = BatchNorm(Filters[1])
    conv2 = Conv((3,3),Filters[1]=>Filters[2], pad = 1, dilation = 1)
    batch2 = BatchNorm(Filters[2])
    conv3 = Conv((1,1),Filters[2]=>Filters[3], pad = 0, dilation = 1)
    batch3 = BatchNorm(Filters[3])

    conv_skip = Conv((3,3), number_of_classes=>Filters[3];
                    stride = 1, pad = 1, dilation = 1)
    batch_skip = BatchNorm(Filters[3])

    #End of first residual block

    #Start of second residual block
    Filters_2 = (64,64,128)
    conv4 = Conv((1,1), 64=>Filters_2[1], pad = 0, dilation = 1)
    batch4 = BatchNorm(Filters_2[1])
    conv5 = Conv((3,3),Filters_2[1]=>Filters_2[2], pad = 1, dilation = 1)
    batch5 = BatchNorm(Filters_2[2])
    conv6 = Conv((1,1),Filters_2[2] => Filters_2[3], pad = 0, dilation = 1)
    batch6 = BatchNorm(Filters_2[3])

    conv_skip_2 = Conv((3,3), 64=>Filters_2[3];
                    stride = 1, pad = 1, dilation = 1)
    batch_skip_2 = BatchNorm(Filters_2[3])

    #End of second residual block

    #Start of third residual block
    Filters_3 = (128,128,256)
    conv7 = Conv((1,1), 128=>Filters_3[1], pad = 0, dilation = 1)
    batch7 = BatchNorm(Filters_3[1])
    conv8 = Conv((3,3),Filters_3[1]=>Filters_3[2], pad = 1, dilation = 1)
    batch8 = BatchNorm(Filters_3[2])
    conv9 = Conv((1,1),Filters_3[2] => Filters_3[3], pad = 0, dilation = 1)
    batch9 = BatchNorm(Filters_3[3])

    conv_skip_3 = Conv((3,3), 128=>Filters_3[3];
        stride = 1, pad = 1, dilation = 1)
    batch_skip_3 = BatchNorm(Filters_3[3])

    #End of third residual block

    #Now we start making the pyramid feature map
    mp1 = MeanPool((1,1); pad = 0, stride = 1)
    ct1 = Conv((1,1), 256 =>64 ,stride = 1, pad = 0, dilation = 1)

    mp2 = MeanPool((2,2); pad = 0, stride = 2)
    ct2 = Conv((1,1),  256=>64, stride = 1, pad = 0, dilation = 1)

    mp3 = MeanPool((4,4); pad = 0, stride = 4)
    ct3 = Conv((1,1),  256=>64 ,stride = 1, pad = 0, dilation = 1)

    mp4 = MeanPool((8,8); pad = 0, stride = 8)
    ct4 = Conv((1,1),  256=>64, stride = 1, pad = 0, dilation = 1)

    #End of feature pyramid

    #Now we make the last convolutional layer which has input from the pyramid feature map plus the fetaures extracted from residual blocks
    #so the output from residual blocks is (256,256,256,1) and the output from the pyramid feature map is four feature maps
    #with shape (256,256,64,1).
    #The last convolutional layer takes input which is concatenation of output from residual blocks and Pyramid feature map
    conv_last = Conv((3,3),512=>number_of_classes, pad = 1, dilation = 1)
    batch_last = BatchNorm(number_of_classes)

end

mutable struct Model
   x_output
   x_skip_output
   x_output_2
   x_skip_output_2
   x_output_3
   x_skip_output_3
   pyramid_feature_1
   pyramid_feature_2
   pyramid_feature_3
   pyramid_feature_4
   last_conv
end

function pspnet(m::PSPNet)

    x_output = Chain(m.conv1, m.batch1, x -> leakyrelu.(x,0.01), m.conv2, m.batch2, x->leakyrelu.(x,0.01), m.conv3, m.batch3, x->leakyrelu.(x,0.01))
    x_skip_output = Chain(m.conv_skip,m.batch_skip)
    x_output_2 = Chain(m.conv4, m.batch4, x->leakyrelu.(x,0.01), m.conv5, m.batch5, x->leakyrelu.(x,0.01), m.conv6, m.batch6, x->leakyrelu.(x,0.01))
    x_skip_output_2 = Chain(m.conv_skip_2,m.batch_skip_2)
    x_output_3 = Chain(m.conv7, m.batch7, x->leakyrelu.(x,0.01), m.conv8, m.batch8, x->leakyrelu.(x,0.01), m.conv9, m.batch9, x->leakyrelu.(x,0.01))
    x_skip_output_3 = Chain(m.conv_skip_3, m.batch_skip_3)
    pyramid_feature_1 = Chain(m.mp1, m.ct1, x->upsample(x,(1,1,1,1)))
    pyramid_feature_2 = Chain(m.mp2, m.ct2, x->upsample(x,(2,2,1,1)))
    pyramid_feature_3 = Chain(m.mp3, m.ct3, x->upsample(x,(4,4,1,1)))
    pyramid_feature_4 = Chain(m.mp4, m.ct4, x->upsample(x,(8,8,1,1)))
    last_conv = Chain(m.conv_last, m.batch_last)
    return Model(x_output, x_skip_output, x_output_2, x_skip_output_2, x_output_3, x_skip_output_3,
                pyramid_feature_1, pyramid_feature_2, pyramid_feature_3, pyramid_feature_4, last_conv)

end

@with_kw mutable struct Args
    batchsize = 10
    λ=0
    η = 0.01
    ρ = 0.99
    seed=0
    epochs = 10
    cuda = true
    checktime = 5        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    savepath = nothing
    train_img_path = joinpath(homedir(), "Downloads", "CityScape_data","TrainImg")  #you can change the path as per your choice
    train_mask_path = joinpath(homedir(), "Downloads", "CityScape_data","TrainMask")
    test_img_path = joinpath(homedir(), "Downloads", "CityScape_data","TestImg")
    test_mask_path = joinpath(homedir(), "Downloads", "CityScape_data","TestMask")
end

function upsample(x,ratio)
  (h, w, c, n) = size(x)
  y = similar(x, (1, ratio[1], 1, ratio[2], 1, 1))
  fill!(y, 1)
  z = reshape(x, (h, 1, w, 1, c, n))  .* y
  reshape(permutedims(z, (2,1,4,3,5,6)), size(x) .* ratio)
end

function forward(feature_extractor::Model,input::Array)
    sum = feature_extractor.x_output(input)+feature_extractor.x_skip_output(input)
    sum2 = feature_extractor.x_output_2(sum)+feature_extractor.x_skip_output_2(sum)
    sum3 = feature_extractor.x_output_3(sum2)+feature_extractor.x_skip_output_3(sum2)
    pf1 = feature_extractor.pyramid_feature_1(sum3)
    pf2 = feature_extractor.pyramid_feature_2(sum3)
    pf3 = feature_extractor.pyramid_feature_3(sum3)
    pf4 = feature_extractor.pyramid_feature_4(sum3)
    matrices = (sum3,pf1,pf2,pf3,pf4)
    concat_matrices = cat(matrices...;dims=(3))
    output = feature_extractor.last_conv(concat_matrices)
    return output
end

rgb_to_array(img)=begin
    H, W = height(img), width(img)
    a = zeros(H, W, 3)
    for w in 1:W
        for h in 1:H
            rgb = img[h,w]
            a[h,w,1] = rgb.r
            a[h,w,2] = rgb.g
            a[h,w,3] = rgb.b
        end
    end
    a
end

function get_data(Train_img_path::String, Train_Label_path::String, Test_img_path::String, Test_Label_path::String)
    Train_data = readdir(Train_img_path, join = true, sort = true)
    Train_Label = readdir(Train_Label_path, join = true, sort = true)
    Test_data = readdir(Test_img_path, join = true, sort = true)
    Test_Label = readdir(Test_Label_path, join = true, sort = true)
    Xtrain = zeros(256,256,3,length(Train_data))
    Ytrain = zeros(256,256,3,length(Train_Label))
    Xtest = zeros(256,256,3,length(Test_data))
    Ytest = zeros(256,256,3,length(Test_Label))
    i,j = 1,1
    for (img_path,mask_path) in zip(Train_data,Train_Label)

        img = load(img_path)
        mask = load(mask_path)
        img = rgb_to_array(img)
        mask = rgb_to_array(mask)
        Xtrain[:,:,:,i] = img
        Ytrain[:,:,:,i] = mask
        i += 1
    end
    for (test_img_path, test_mask_path) in zip(Test_data,Test_Label)
        test_img = load(test_img_path)
        test_mask = load(test_mask_path)
        img = rgb_to_array(test_img)
        mask = rgb_to_array(test_mask)
        Xtest[:,:,:,j] = img
        Ytest[:,:,:,j] = mask
        j += 1
    end

    train_loader = DataLoader(Xtrain,Ytrain,batchsize = args.batchsize,shuffle = false)
    test_loader = DataLoader(Xtest,Ytest,batchsize = args.batchsize,shuffle = false)

    return train_loader, test_loader
end

function train(psp_layers::PSPNet, Args::Args)
    args = Args()
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.cuda && CUDAapi.has_cuda_gpu()
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    if args.savepath == nothing
     experiment_folder = savename("PSPNet", args, scientific=3,
                 accesses = [:batchsize, :η, :λ]) # construct path from these fields
     args.savepath = joinpath("runs", experiment_folder)
    end

    ## DATA
    train_loader, test_loader = get_data(args.train_img_path , args.train_mask_path , args.test_img_path , args.test_mask_path)
    @info "Dataset CityScape: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

    ## MODEL AND OPTIMIZER
    layers = psp_layers
    model = pspnet(layers)|> device
    ps = Flux.params(model.x_output , model.x_skip_output , model.x_output_2 , model.x_skip_output_2 , model.x_output_3 , model.x_skip_output_3 ,
                    model.pyramid_feature_1 , model.pyramid_feature_2 , model.pyramid_feature_3 , model.pyramid_feature_4 ,
                    model.last_conv)
    loss(x , y) = Flux.mse(x , y)
    opt = ADAM(args.η)
    if args.λ > 0
        opt = Optimiser(opt , WeightDecay(args.λ))
    end

    ## TRAINING
    @info "Start Training"
    report(0)
    for epoch in 1:args.epochs
        p = ProgressMeter.Progress(length(train_loader))

        for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                ŷ = forward(model,x)
                loss(ŷ, y)
            end
            Flux.Optimise.update!(opt, ps, gs)
            ProgressMeter.next!(p)   # comment out for no progress bar
        end

        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)
            modelpath = joinpath(args.savepath, "model.bson")
            let model=cpu(model), args=struct2dict(args)
                BSON.@save modelpath model epoch args
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
end
