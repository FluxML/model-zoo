# # Flux Transformer

# This is a walkthrough of the Transformer network.
# We'll start by taking a look at the data we'll use.

corpus(name) = split(String(read(name)), "\n")

english = corpus(joinpath(@__DIR__, "data", "europarl-v7.fr-en.en"))
french  = corpus(joinpath(@__DIR__, "data", "europarl-v7.fr-en.fr"))

english[1000]
french[1000]

# Our corpus is conveniently already split into sentences, but we need each
# sentence to be further split into words. This is easy with WordTokenizers.

using WordTokenizers
tokenize("I don't know")

using DelimitedFiles

dict_en = string.(vec(readdlm(joinpath(@__DIR__, "data", "dict.en"))))
dict_fr = string.(vec(readdlm(joinpath(@__DIR__, "data", "dict.fr"))))
push!(dict_en, "UNK"); push!(dict_fr, "UNK")

using Flux
using Flux: onehotbatch

encode(sentence, dict) = onehotbatch(lowercase.(tokenize(sentence)), dict, "UNK")

encode("I don't know", dict_en)

N = 512

embedding = param(rand(N, length(dict_en)))

embedding * encode("I don't know", dict_en)
