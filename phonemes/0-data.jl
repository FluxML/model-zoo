using Flux, Flux.Batches, CMUDict
using Flux: onehot
using Base.Iterators: partition

cd(@__DIR__)

# Maximum length of a sequence
Nseq = 35
Nbatch = 50

# Augment the alphabet with a "stop" token.
alphabet = ['\0', CMUDict.alphabet...]
phones = [:end, CMUDict.symbols...]

# One-hot encode each letter, turn into a sequence, pad with `\0`s.
function tokenise(s, alphabet)
  rpad(Seq([onehot(Float32, ch, alphabet) for ch in s]),
       Nseq, onehot(Float32, alphabet[1], alphabet))
end

# Turn a word into a sequence of vectors
tokenise("PHYLOGENY", alphabet)
# See the raw data
rawbatch(tokenise("PHYLOGENY", alphabet))
# Same for phoneme lists
tokenise(CMUDict.dict["PHYLOGENY"], phones)

# Finally, create iterators for our inputs and outputs.
Xs = (Batch([xs...]) for xs in
      partition((tokenise(word, alphabet)
                 for word in keys(CMUDict.dict)),
                Nbatch))

Ys = (Batch([xs...]) for xs in
      partition((tokenise(CMUDict.dict[word], phones)
                 for word in keys(CMUDict.dict)),
                Nbatch))

# Peek at the first batch
first(Xs)
first(Ys)

# You can collect the generators if you have about 4GiB free
# Xs, Ys = collect(Xs), collect(Ys)
