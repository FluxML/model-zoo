using Flux, Flux.Batches, CMUDict
using Flux: onehot
using Flux.Batches: batches

cd(@__DIR__)

# Maximum length of a sequence
Nseq = 35
Nbatch = 50

alphabet = [:end, CMUDict.alphabet...]
phones = [:start, :end, CMUDict.symbols...]

# One-hot encode each letter, turn into a sequence, pad with `\0`s.
function tokenise(s, alphabet)
  rpad(Seq([onehot(Float32, ch, alphabet) for ch in s]),
       Nseq, onehot(Float32, :end, alphabet))
end

# Turn a word into a sequence of vectors
tokenise("PHYLOGENY", alphabet)
# See the raw data
rawbatch(tokenise("PHYLOGENY", alphabet))
# Same for phoneme lists
tokenise(CMUDict.dict["PHYLOGENY"], phones)

words = collect(keys(CMUDict.dict))

# # Finally, create iterators for our inputs and outputs.
Xs = batches((tokenise(word, alphabet) for word in words),
             Nbatch)

Ys = batches((tokenise(CMUDict.dict[word], phones) for word in words),
             Nbatch)

Yoffset = batches((tokenise([:start, CMUDict.dict[word]...], phones) for word in words),
                  Nbatch)

# Peek at the first batch
first(Xs)
first(Ys)

# You can collect the generators if you have about 4GiB free
# Xs, Ys = collect(Xs), collect(Ys)
