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

function getindex(data::ImageContainer, idx::Int)
    path = data.img[idx]
    name = replace(path, r"(.+)\\(.+)\\(.+_\d+)\.jpg" => s"\2")    
    img = Images.load(path)
    img = apply(tfm, Image(img))
    img = permutedims(channelview(RGB.(itemdata(img))), (3, 2, 1))
    img = Float32.(img)
    y = name_to_idx[name]
    return img, Flux.onehotbatch(y, 1:3)
end

# define DataLoaders
const batchsize = 16

dtrain = Flux.DataLoader(
    ImageContainer(imgs[1:2700]);
    batchsize,
    collate = true,
    parallel = true
)
device == gpu ? dtrain = Flux.CuIterator(dtrain) : nothing

deval = Flux.DataLoader(
    ImageContainer(imgs[2701:3000]);
    batchsize,
    collate = true,
    parallel = true
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
        good += sum(Flux.onecold(m(x)) .== Flux.onecold(y))
        count += size(y, 2)
    end
    acc = round(good / count, digits = 4)
    return acc
end

function train_epoch!(m; ps, opt, dtrain)
    for (x, y) in dtrain
        grads = gradient(ps) do
            Flux.Losses.logitcrossentropy(m(x), y)
        end
        update!(opt, ps, grads)
    end
end

ps = Flux.params(m_tot[2:end]);
opt = Adam(3e-4)

for iter = 1:8
    @time train_epoch!(m_tot; ps, opt, dtrain)
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
        good += sum(Flux.onecold(m_tune(m_infer(x))) .== Flux.onecold(y))
        count += size(y, 2)
    end
    acc = round(good / count, digits = 4)
    return acc
end

function train_epoch!(m_infer, m_tune; ps, opt, dtrain)
    for (x, y) in dtrain
        infer = m_infer(x)
        grads = gradient(ps) do
            Flux.Losses.logitcrossentropy(m_tune(infer), y)
        end
        update!(opt, ps, grads)
    end
end

ps = Flux.params(m_tune);
opt = Adam(3e-4)

# training loop
for iter = 1:8
    @time train_epoch!(m_infer, m_tune; ps, opt, dtrain)
    metric_train = eval_f(m_infer, m_tune, dtrain)
    metric_eval = eval_f(m_infer, m_tune, deval)
    @info "train" metric = metric_train
    @info "eval" metric = metric_eval
end
