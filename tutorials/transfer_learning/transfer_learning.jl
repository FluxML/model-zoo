# load packages
using Random: shuffle!
import Base: length, getindex
using Images
using Flux
using Flux: update!
using DataAugmentation
using Metalhead

device = Flux.CUDA.functional() ? gpu : cpu
# device = cpu

## Custom DataLoader
const CATS = readdir(abspath(joinpath("data", "animals", "cats")), join = true)
const DOGS = readdir(abspath(joinpath("data", "animals", "dogs")), join = true)
const PANDA = readdir(abspath(joinpath("data", "animals", "panda")), join = true)

struct ImageContainer{T<:Vector}
    img::T
end

imgs = [CATS..., DOGS..., PANDA...]
shuffle!(imgs)
data = ImageContainer(imgs)

length(data::ImageContainer) = length(data.img)

const im_size = (224, 224)
tfm = DataAugmentation.compose(ScaleKeepAspect(im_size), CenterCrop(im_size))
name_to_idx = Dict{String,Int32}("cats" => 1, "dogs" => 2, "panda" => 3)

const mu = [0.485f0, 0.456f0, 0.406f0]
const sigma = [0.229f0, 0.224f0, 0.225f0]

function getindex(data::ImageContainer, idx::Int)
    path = data.img[idx]
    _img = Images.load(path)
    _img = itemdata(apply(tfm, Image(_img)))
    img = collect(channelview(float32.(RGB.(_img))))
    img = permutedims((img .- mu) ./ sigma, (3, 2, 1))
    name = replace(path, r"(.+)\\(.+)\\(.+_\d+)\.jpg" => s"\2")
    y = name_to_idx[name]
    return img, y
end

# define DataLoaders
const batchsize = 16

dtrain = Flux.DataLoader(
    ImageContainer(imgs[1:2700]);
    batchsize,
    collate = true,
    parallel = true,
)
device == gpu ? dtrain = Flux.CuIterator(dtrain) : nothing

deval = Flux.DataLoader(
    ImageContainer(imgs[2701:3000]);
    batchsize,
    collate = true,
    parallel = true,
)
device == gpu ? deval = Flux.CuIterator(deval) : nothing

# Fine-tune | 🐢 mode
# Load a pre-trained model: 
m = Metalhead.ResNet(18, pretrain = true).layers
m_tot = Chain(m[1], AdaptiveMeanPool((1, 1)), Flux.flatten, Dense(512 => 3)) |> device

function eval_f(m, deval)
    good = 0
    count = 0
    for (x, y) in deval
        good += sum(Flux.onecold(m(x)) .== y)
        count += length(y)
    end
    acc = round(good / count, digits = 4)
    return acc
end

function train_epoch!(model; opt, dtrain)
    for (x, y) in dtrain
        grads = gradient(model) do m
            Flux.Losses.logitcrossentropy(m(x), Flux.onehotbatch(y, 1:3))
        end
        update!(opt, model, grads[1])
    end
end

opt = Flux.setup(Flux.Optimisers.Adam(1e-5), m_tot);

for iter = 1:5
    @time train_epoch!(m_tot; opt, dtrain)
    metric_train = eval_f(m_tot, dtrain)
    metric_eval = eval_f(m_tot, deval)
    @info "train" metric = metric_train
    @info "eval" metric = metric_eval
end

# Fine-tune | 🐇 mode
# define models 
m_infer = deepcopy(m[1]) |> device
m_tune = Chain(AdaptiveMeanPool((1, 1)), Flux.flatten, Dense(512 => 3)) |> device

function eval_f(m_infer, m_tune, deval)
    good = 0
    count = 0
    for (x, y) in deval
        good += sum(Flux.onecold(m_tune(m_infer(x))) .== y)
        count += length(y)
    end
    acc = round(good / count, digits = 4)
    return acc
end

function train_epoch!(m_infer, m_tune; opt, dtrain)
    for (x, y) in dtrain
        infer = m_infer(x)
        grads = gradient(m_tune) do m
            Flux.Losses.logitcrossentropy(m(infer), Flux.onehotbatch(y, 1:3))
        end
        update!(opt, m_tune, grads[1])
    end
end

opt = Flux.setup(Flux.Optimisers.Adam(1e-3), m_tune);

# training loop
for iter = 1:5
    @time train_epoch!(m_infer, m_tune; opt, dtrain)
    metric_train = eval_f(m_infer, m_tune, dtrain)
    metric_eval = eval_f(m_infer, m_tune, deval)
    @info "train" metric = metric_train
    @info "eval" metric = metric_eval
end
