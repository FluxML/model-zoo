#-------Imports-------
using Flux, Images, Statistics
using CuArrays, CUDAnative
using BSON: @save
using Flux: @interrupts, @epochs, throttle, train!

#-------Utilities-------
function center_crop(x, target_size_h, target_size_w)
    start_h = (size(x, 1) - target_size_h) ÷ 2 + 1
    start_w = (size(x, 2) - target_size_w) ÷ 2 + 1
    x[start_h:(start_h + target_size_h - 1), start_w:(start_w + target_size_w - 1), :, :]
end

#-------UNet Architecture-------
UNetConvBlock(in_chs, out_chs, kernel = (3, 3)) =
    Chain(Conv(kernel, in_chs=>out_chs, relu, pad = (1, 1)),
          Conv(kernel, out_chs=>out_chs, relu, pad = (1, 1)))

struct UNetUpBlock
    upsample
    conv_layer
end

@treelike UNetUpBlock

UNetUpBlock(in_chs::Int, out_chs::Int, kernel = (3, 3)) =
    UNetUpBlock(ConvTranspose((2, 2), in_chs=>out_chs, stride=(2, 2)),
                Chain(Conv(kernel, in_chs=>out_chs, relu, pad=(1, 1)),
                      Conv(kernel, out_chs=>out_chs, relu, pad=(1, 1))))

function (u::UNetUpBlock)(x, bridge)
    x = u.upsample(x)
    # Since we know the image dimensions from beforehand we might as well not use the center_crop
    # u.conv_layer(cat(x, center_crop(bridge, size(x, 1), size(x, 2)), dims = 3))
    u.conv_layer(cat(x, bridge, dims = 3))
end

struct UNet
    pool_layer
    conv_blocks
    up_blocks
end

@treelike UNet

# This is to be used for Background and Foreground segmentation
function UNet()
    pool_layer = MaxPool((2, 2))
    conv_blocks = (UNetConvBlock(3, 64), UNetConvBlock(64, 128), UNetConvBlock(128, 256),
                   UNetConvBlock(256, 512), UNetConvBlock(512, 1024))
    up_blocks = (UNetUpBlock(1024, 512), UNetUpBlock(512, 256), UNetUpBlock(256, 128),
                 UNetUpBlock(128, 64), Conv((1, 1), 64=>1))
    UNet(pool_layer, conv_blocks, up_blocks)
end

function (u::UNet)(x)
    outputs = Vector(undef, 5)
    outputs[1] = u.conv_blocks[1](x)
    for i in 2:5
        pool_x = u.pool_layer(outputs[i - 1])
        outputs[i] = u.conv_blocks[i](pool_x)
    end
    up_x = outputs[end]
    for i in 1:4
        up_x = u.up_blocks[i](up_x, outputs[end - i])
    end
    u.up_blocks[end](up_x)
end

#-------Dataset Loading-------
function get_img_paths(data_dir)
    img_path = []
    for i in readdir(data_dir)
        if split(i, ".")[end] == "jpg" || split(i, ".")[end] == "gif"
            push!(img_path, joinpath(data_dir, i))
        end
    end
    img_path
end

loadim(path) = float.(permutedims(channelview(imresize(load(path), 224, 224)), (2, 3, 1)))
loadmask(path) = float.(channelview(imresize(load(path), 224, 224)))[1, :, :]

train_img_paths = get_img_paths("train")
train_mask_paths = get_img_paths("train_masks")
train_imgs = []
train_masks = []
dices = []
losses = []

# Since there are only a few images we can load the entire dataset at once
function load_train_imgs()
    global train_imgs
    imgs_loaded = 0
    for i in train_img_paths
        push!(train_imgs, loadim(i))
        imgs_loaded += 1
        if imgs_loaded % 1000 == 0
            @info "$imgs_loaded Images have been loaded"
        end
    end
end

# Since there are only a few images we can load the entire dataset at once
function load_train_masks()
    global train_masks
    masks_loaded = 0
    for i in train_mask_paths
        push!(train_masks, loadmask(i))
        masks_loaded += 1
        if masks_loaded % 1000 == 0
            @info "$masks_loaded Masks have been loaded"
        end
    end
end

#-------Creating the model-------
batch_size = 16
n_epochs = 5
model = UNet() |> gpu
opt = ADAM()
load_train_imgs()
load_train_masks()

#-------Loss and Metric Functions-------
function dice_coeff(x, y)
    ŷ = model(x)
    val = 2 * sum(ŷ .* y) / (sum(ŷ) + sum(y))
    @info "Dice Coefficient = $(val)"
end               

function logsig(x)
    max_v = max.(zero.(x), -x)
    z = CUDAnative.exp.(-max_v) + CUDAnative.exp.(-(x .+ max_v))
    -(max_v .+ CUDAnative.log.(z))
end

logitbinarycrossentropy(logŷ, y) = (1 .- y) .* logŷ .- logsig(logŷ)

loss(x, y) = mean(logitbinarycrossentropy.(model(x |> gpu), y |> gpu))

#-------Training the model-------

train_dataset = [(cat(train_imgs[i]..., dims = 4), cat(train_masks[i]..., dims = 4))
                 for i in partition(1:(length(train_imgs) - batch_size), batch_size)]

val_dataset = (cat(train_imgs[(length(train_imgs)- batch_size + 1):end]..., dims = 4),
               cat(train_masks[(length(train_imgs)- batch_size + 1):end]..., dims = 4))

evalcb = throttle(() -> dice_coeff(val_dataset[1] |> gpu, val_dataset[2] |> gpu), 60)

@epochs n_epochs train!(loss, params(model), train_dataset, opt, cb = evalcb)

@save "unet.bson" model |> cpu
