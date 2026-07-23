using Flux: chunk, batchseq
using Base.Iterators: partition
using Downloads: download

function load_and_clean_data()
    isfile("train.txt") ||
            download("http://classics.mit.edu/Plato/republic.mb.txt", "train.txt")

    # read string
    doc = lowercase(String(read("train.txt")))

    # prepare corpus for tokenizer
    doc = replace(
        doc,
        # replace common contractions
        "n't" => " not",
        "'s" => 's',
        # replace chapter dividers with white space
        "--" => ' ',
        # replace quotes with white space
        '\'' => ' ',
        '"' => ' '
    )

    # remove end of sentence punctuation
    tokens = replace.(split(doc), r"(?<!\.)[.;,?!]\z" => "")

    # keep only alphabetic tokens
    filter!(w -> all(isletter(c) for c in w), tokens)

    return tokens
end

function get_tokens_and_vocabulary()
    tokens = load_and_clean_data()

    ##### borrowed from model-zoo/text/treebank/data.jl #####
    # Count how many times each token appears.
    freqs = Dict{String,Int}()
    for t in tokens
      freqs[t] = get(freqs, t, 0) + 1
    end

    # Replace singleton tokens with an "unknown" marker.
    # This roughly cuts our "vocabulary" of tokens in half.
    tokens = replace(t -> get(freqs, t, 0) == 1 ? "UNK" : t, tokens)
    ##########

    # create vocabulary
    vocabulary = unique(tokens)

    return tokens, vocabulary
end

function onehot_data(batch, labels)
    # create targets with dimension (sequence_length x vocab_size x samples)
    return [[Flux.onehotbatch(b_i, labels) for b_i in b] for b in batch]
end

function batchify_data(tokens, unk_token, args)
    # restructure data into batches of dimension sequence_length x (features x samples)
    return batchseq.(partition.(chunk(tokens, args.nbatch), args.seqlen), unk_token)
end

function get_data(args)
    # load the raw data
    tokens, vocabulary = get_tokens_and_vocabulary()

    # vocab_size calculated from corpus
    vocab_size = length(vocabulary)

    # map words to their indices in vocabulary array
    word2ind = Dict(vocabulary .=> 1:vocab_size)

    # unknown token in vocabulary
    unk = word2ind["UNK"]

    # convert string tokens to integers
    tokens = map(x -> get(word2ind, x, nothing), tokens)

    # final data format
    x_train = batchify_data(tokens[1:end-1], unk, args)
    y_train = onehot_data(batchify_data(tokens[2:end], unk, args), 1:vocab_size)

    return x_train, y_train, word2ind, vocabulary
end
