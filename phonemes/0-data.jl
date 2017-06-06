using Flux, Flux.Batches, CMUDict
using Flux: onehot
using Base.Iterators: partition

# Maximum length of a sequence
Nseq = 35
Nbatch = 50

# Augment the alphabet with a "stop" token.
alphabet = ['\0', CMUDict.alphabet...]
phones = [:end, CMUDict.symbols...]

# One-hot encode each letter, turn into a sequence, pad with `\0`s.
function tokenize(s, alphabet)
  rpad(Seq([onehot(Float32, ch, alphabet) for ch in s]),
       Nseq, onehot(Float32, alphabet[1], alphabet))
end

# Turn a word into a sequence of vectors
tokenize("PHYLOGENY", alphabet)
# See the raw data
rawbatch(tokenize("PHYLOGENY", alphabet))
# Same for phoneme lists
tokenize(CMUDict.dict["PHYLOGENY"], phones)

# Finally, create iterators for our inputs and outputs.
Xs = (Batch([xs...]) for xs in
      partition((tokenize(word, alphabet)
                 for word in keys(CMUDict.dict)),
                Nbatch))

Ys = (Batch([xs...]) for xs in
      partition((tokenize(CMUDict.dict[word], phones)
                 for word in keys(CMUDict.dict)),
                Nbatch))

# Peek at the first batch
first(Xs)
first(Ys)
