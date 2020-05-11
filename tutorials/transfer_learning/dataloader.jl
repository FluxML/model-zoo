using Flux, Images
using Random
using StatsBase: sample, shuffle
using Base.Iterators

const PATH = joinpath(@__FILE__, "train")
const FILES = joinpath.(PATH, readdir(PATH))
if isempty(readdir(PATH))
  error("Empty train folder - perhaps you need to download and extract the kaggle dataset.")
end

const DOGS = filter(x -> occursin("dog", x), FILES)
const CATS = filter(x -> occursin("cat", x), FILES)

function load_batch(n = 10, nsize = (224,224); path = PATH)
net_chain[1:end-1]
  imgs_paths = shuffle(vcat(sample(DOGS, n/2), sample(CATS, n/2)))
  labels = map(x -> occursin("dog",x) ? 1 : 0, imgs_path)
  labels = Flux.onehotbatch(labels, [0,1])
  imgs = Images.load.(imgs_path)
  imgs = map(img -> Images.imresize(img, nsize...), imgs)
  imgs = map(img -> permutedims(channelview(img), (3,2,1)), imgs)
  imgs = cat(imgs..., dims = 4)
  imgs, labels
end
