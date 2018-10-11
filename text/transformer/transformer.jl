# # The Transformer

# This script is a walkthrough of the Transformer network.
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
