# Transfer learning of vision model with Flux

## Context

This tutorial shows how to perform transfer learning using a pre-trained vision model. In the process, we will also learn how to use a custom data container, a useful feature when dealing with large datasets that cannot fit into memory.

Transfer learning is an common way in which large, compute intensive models can be used in practice. Following their training to perform well on their general trask, they can be subsequently used as a basis to fine-tune only some of their components on smaller, specialized datasets adapted to a specific task.

Self contained Julia code presented in this tutorial is found in ["transfer_learning.jl"](transfer_learning.jl) and can be launched with: 

```
julia project=@. --threads=8 transfer_learning.jl
```

## Getting started

In this tutorial, we'll used a pre-trained ResNet18 model to solve a 3-class classification problem: 🐱, 🐶, 🐼.

Data can be accessed from [Kaggle](https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda).

Following download and unzip, data is expected to live under the following structure: 

```
- data
    - animals
        - cats
        - dogs
        - panda
```

In Julia, the following packages are needed:

```julia
using Random: shuffle!
import Base: length, getindex
using Images
using Flux
using Flux: update!
using DataAugmentation
using Metalhead
```

We also define a utility to help manage cpu-gpu conversion:

```julia
device = Flux.CUDA.functional() ? gpu : cpu
```

A CUDA enabled GPU is recommended. A modest one with 6GB RAM is sufficient and should run the tutorial in just 5-6 mins. If running on CPU, it may take over 40 mins. 

## Custom DataLoader

When dealing with large datasets, it's unrealistic to use a vanilla `DataLoader` constructor using the entire dataset as input. A handy approach is to rely on custom data containers, which allows to only pull data into memory as needed. 

Our custom data container is very simple. It's a `struct` containing the paths to each of our images: 

```julia
const CATS = readdir(abspath(joinpath("data", "animals", "cats")), join = true)
const DOGS = readdir(abspath(joinpath("data", "animals", "dogs")), join = true)
const PANDA = readdir(abspath(joinpath("data", "animals", "panda")), join = true)

struct ImageContainer{T<:Vector}
    img::T
end

imgs = [CATS..., DOGS..., PANDA...]
shuffle!(imgs)
data = ImageContainer(imgs)
```

In order to be compatible with `DataLoader`, 2 functions must minimally be defined:
 - `Base.length`: returns the number of observations in the data container.
 - `Base.getindex`: function that returns the observation for a specified index. 

```julia
length(data::ImageContainer) = length(data.img)

const im_size = (224, 224)
tfm = DataAugmentation.compose(ScaleKeepAspect(im_size), CenterCrop(im_size))
name_to_idx = Dict{String,Int32}("cats" => 1, "dogs" => 2, "panda" => 3)

function getindex(data::ImageContainer, idx::Int)
    path = data.img[idx]
    img = Images.load(path)
    img = apply(tfm, Image(img))
    img = permutedims(channelview(RGB.(itemdata(img))), (3, 2, 1))
    img = Float32.(img)
    name = replace(path, r"(.+)\\(.+)\\(.+_\d+)\.jpg" => s"\2")    
    y = name_to_idx[name]
    return img, Flux.onehotbatch(y, 1:3)
end
```

In the above, the class label `y` is obtained by first using a regexp to extract the parent folder name of the image, which can be one of `cats`, `dogs` or `panda`. Then, this name can be mapped into an integer index using the `name_to_idx` dictionary. 

Data augmentation is performed through the `tfm` pipeline powered by [DataAugmentation.jl]https://github.com/lorenzoh/DataAugmentation.jl. Random crops, flips and color augmentation techniques are also supported.  

We can now define our train and eval data iterators: 

```julia
const batchsize = 16

dtrain = Flux.DataLoader(
    ImageContainer(imgs[1:2700]);
    batchsize,
    collate = true,
    parallel = true
)
device == gpu ? dtrain = Flux.CuIterator(dtrain) : nothing
```

```julia
deval = Flux.DataLoader(
    ImageContainer(imgs[2701:3000]);
    batchsize,
    collate = true,
    parallel = true
)
device == gpu ? deval = Flux.CuIterator(deval) : nothing
```

The `collate` option is set to `true` in order for all of the images to be concatenated into a 4D Array, where the batch dimension is last. Is set to false, it will return a vector of length `batchsize`, in which each element is a single 3D Array (width, height, channels).

Setting `parallel` to `true` is an important performance enhancement as the GPU would otherwise spent significant time on idle waiting for the CPU data loading to complete. 

## Fine-tune | 🐢 mode

Load a pre-trained model: 

```julia
m = Metalhead.ResNet(18, pretrain = true).layers
```

Substitute the latest layers with ones adapted to the fine-tuning task:

```julia
m_tot = Chain(m[1], AdaptiveMeanPool((1, 1)), Flux.flatten, Dense(512 => 3)) |> device
```

Define an accuracy evaluation function:

```julia
function eval_f(m, deval)
    good = 0
    count = 0
    for (x, y) in deval
        good += sum(Flux.onecold(m(x)) .== Flux.onecold(y))
        count += size(y, 2)
    end
    acc = round(good / count, digits = 4)
    return acc
end
```

Define a training loop for 1 epoch: 

```julia
function train_epoch!(m; ps, opt, dtrain)
    for (x, y) in dtrain
        grads = gradient(ps) do
            Flux.Losses.logitcrossentropy(m(x), y)
        end
        update!(opt, ps, grads)
    end
end
```

Set learnable parameters and optimiser:

```julia
ps = Flux.params(m_tot[2:end]);
opt = Adam(3e-4)
```

Train for a few epochs:

```julia
for iter = 1:8
    @time train_epoch!(m_tot; ps, opt, dtrain)
    metric_train = eval_f(m_tot, dtrain)
    metric_eval = eval_f(m_tot, deval)
    @info "train" metric = metric_train
    @info "eval" metric = metric_eval
end
 12.932525 seconds (3.01 M allocations: 7.001 GiB, 16.11% gc time)
┌ Info: train
└   metric = 0.9481
┌ Info: eval
└   metric = 0.9467
```

## Fine-tune | 🐇 mode

In the previous fine-tuning, despite having only specified the last `Dense` layer as trainable parameters, we nonetheless ended computing the gradients over the entire model. 

To avoid these unnecessary computations, we can split our model in two: 
- The original pre-trained core, for which we don't want to compute gradients
- The new final layers, for which gradients are needed. 

```julia
m_infer = deepcopy(m[1]) |> device
m_tune = Chain(AdaptiveMeanPool((1, 1)), Flux.flatten, Dense(512 => 3)) |> device
```

Only minimal adaptations are then needed to the eval and training functions: 

```julia
function eval_f(m_infer, m_tune, deval)
    good = 0
    count = 0
    for (x, y) in deval
        good += sum(Flux.onecold(m_tune(m_infer(x))) .== Flux.onecold(y))
        count += size(y, 2)
    end
    acc = round(good / count, digits = 4)
    return acc
end
```

```julia
function train_epoch!(m_infer, m_tune; ps, opt, dtrain)
    for (x, y) in dtrain
        infer = m_infer(x)
        grads = gradient(ps) do
            Flux.Losses.logitcrossentropy(m_tune(infer), y)
        end
        update!(opt, ps, grads)
    end
end
```

```julia
ps = Flux.params(m_tune);
opt = Adam(3e-4)
```

```julia
for iter = 1:8
    @time train_epoch!(m_infer, m_tune; ps, opt, dtrain)
    metric_train = eval_f(m_infer, m_tune, dtrain)
    metric_eval = eval_f(m_infer, m_tune, deval)
    @info "train" metric = metric_train
    @info "eval" metric = metric_eval
end
  5.470515 seconds (1.33 M allocations: 6.911 GiB, 29.60% gc time)
┌ Info: train
└   metric = 0.9322
┌ Info: eval
└   metric = 0.9467
```

As we can see, an over 2X speedup can be achieved by avoiding unneeded gradients calculations.

**This concludes the tutorial, happy transfer!**