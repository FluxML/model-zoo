# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# %%
using Flux, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Metalhead:trainimgs, CIFAR10
using Images


# %%
using Flatten


# %%
getarray(X) = Float32.(permutedims(channelview(X), (2, 3, 1)))

# %% [markdown]
# ## CIFAR 10 dataset

# %%
X = trainimgs(CIFAR10)
imgs = [getarray(X[i].img) for i in 1:50000];
labels = onehotbatch([X[i].ground_truth.class for i in 1:50000],1:10);
train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:49000, 100)]);


# %%
valset = collect(49001:50000)
valX = cat(imgs[valset]..., dims = 4) |> gpu
valY = labels[:, valset] |> gpu

# %% [markdown]
# ## A block of Conv Relu Batchnorm based on input and output channels

# %%
conv_block(in_channels, out_channels) = (
    Conv((3,3), in_channels => out_channels, relu, pad = (1,1), stride = (1,1)), 
    BatchNorm(out_channels))

# %% [markdown]
# ## Two of the conv blocks which is common in VGG + Maxpool

# %%
double_conv(in_channels, out_channels) = (
    conv_block(in_channels, out_channels),
    conv_block(out_channels, out_channels),
    MaxPool((2,2)))

# %% [markdown]
# ## Modified with 2 conv blocks, 1 conv and max pool

# %%
triple_conv(in_channels, out_channels) = (
    conv_block(in_channels, out_channels),
    conv_block(out_channels, out_channels),
    Conv((3,3), out_channels => out_channels, relu, pad = (1,1), stride = (1,1)),
    MaxPool((2,2)))

# %% [markdown]
# ## VGG
# - ... operator will help us unroll the previously defined blocks

# %%
vgg19(initial_channels, num_classes) = Chain(
    double_conv(initial_channels, 64)...,
    double_conv(64, 128)...,
    conv_block(128,256),
    triple_conv(256,256)...,
    conv_block(256,512),
    triple_conv(512,512)...,
    conv_block(512,512),
    triple_conv(512,512)...,
    x -> reshape(x, :, size(x, 4)),
    Dense(512, 4096, relu),
    Dropout(0.5),
    Dense(4096, 4096, relu),
    Dropout(0.5),
    Dense(4096, num_classes),
    softmax) |> gpu


# %%
m = vgg19(3, 10)


# %%
loss(x, y) = crossentropy(m(x), y)


# %%
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))


# %%
evalcb = throttle(() -> @show(accuracy(valX, valY)), 10)


# %%
opt = ADAM()


# %%
Flux.train!(loss, params(m), train, opt, cb = evalcb)


# %%



