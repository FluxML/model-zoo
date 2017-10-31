# Pkg.clone("https://github.com/FluxML/CMUDict.jl")
# Pkg.build("CMUDict")

using Flux, CMUDict
using Flux: onehot, batchseq
using Base.Iterators: partition

alphabet = [:end, CMUDict.alphabet...]
phones = [:start, :end, CMUDict.symbols...]

tokenise(s, α) = [onehot(c, α) for c in s]

# Turn a word into a sequence of vectors
tokenise("PHYLOGENY", alphabet)
# Same for phoneme lists
tokenise(cmudict["PHYLOGENY"], phones)

words = sort(collect(keys(cmudict)), by = length)

# Finally, create iterators for our inputs and outputs.
batches(xs, p) = [batchseq(b, p) for b in partition(xs, 50)]

Xs = batches([tokenise(word, alphabet) for word in words],
             onehot(:end, alphabet))

Ys = batches([tokenise([cmudict[word]..., :end], phones) for word in words],
             onehot(:end, phones))

Yo = batches([tokenise([:start, cmudict[word]...], phones) for word in words],
             onehot(:end, phones))

data = collect(zip(Xs, Yo, Ys))
