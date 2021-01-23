using Flux, Images
using StatsBase: sample, shuffle

const PATH = joinpath(@__DIR__, "train")
const FILES = joinpath.(PATH, readdir(PATH))
if isempty(readdir(PATH))
  error("Empty train folder - perhaps you need to download and extract the kaggle dataset.")
end

const DOGS = filter(x -> occursin("dog", x), FILES)
const CATS = filter(x -> occursin("cat", x), FILES)

function load_batch(n = 10, nsize = (224,224); path = PATH)
  imgs_paths = shuffle(vcat(sample(DOGS, Int(n/2)), sample(CATS, Int(n/2))))
  labels = map(x -> occursin("dog.",x) ? 1 : 0, imgs_paths)
  labels = Flux.onehotbatch(labels, [0,1])
  imgs = Images.load.(imgs_paths)
  imgs = map(img -> Images.imresize(img, nsize...), imgs)
  imgs = map(img -> permutedims(channelview(img), (3,2,1)), imgs)
  imgs = cat(imgs..., dims = 4)
  Float32.(imgs), labels
end
