using Flux
using Flux: onehot
using Flux.Data.Sentiment
using Flux.Data: Tree, leaves

function getdata()
    traintrees = Sentiment.train()

    ## Get the raw labels and phrases as separate trees.
    labels  = map.(x -> x[1], traintrees)
    phrases = map.(x -> x[2], traintrees)

    ## All tokens in the training set.
    tokens = vcat(map(leaves, phrases)...)

    ## Count how many times each token appears.
    freqs = Dict{String,Int}()
    for t in tokens
      freqs[t] = get(freqs, t, 0) + 1
    end

    ## Replace singleton tokens with an "unknown" marker.
    ## This roughly cuts our "alphabet" of tokens in half.
    phrases = map.(t -> get(freqs, t, 0) == 1 ? "UNK" : t, phrases)

    ## Our alphabet of tokens.
    alphabet = unique(vcat(map(leaves, phrases)...))

    ## One-hot-encode our training data with respect to the alphabet.
    phrases_e = map.(t -> t == nothing ? t : onehot(t, alphabet), phrases)
    labels_e  = map.(t -> onehot(t, 0:4), labels)

    train = map.(tuple, phrases_e, labels_e)
    return train, alphabet
end
