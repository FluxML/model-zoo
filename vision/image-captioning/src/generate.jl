using Pkg
Pkg.activate(".")
using JSON
using WordTokenizers
using StatsBase
using Flux,CuArrays
using Flux:onehot
using Base.Iterators:partition
using Metalhead
using JLD
using BSON:@save,@load

include("utils.jl")
BASE_PATH = "../../references/mscoco/"
NUM_SENTENCES = 30000
K = 5000

# Load the models
@load "cnn_encoder.bson" cnn_encoder
@load "embedding.bson" embedding
@load "rnn_decoder.bson" rnn_decoder
@load "decoder.bson" decoder

vgg = VGG19() |> gpu
Flux.testmode!(vgg)
vgg = vgg.layers[1:end-3] |> gpu

word2idx = load("word2idx.jld")["word2idx"]
idx2word = load("idx2word.jld")["idx2word"]

onehotword(word) = Float32.(onehot(word2idx[word],1:length(keys(word2idx))))

function reset(rnn_decoder)
    Flux.reset!(rnn_decoder)
end

# Call this function with the path to the image to get the caption returned as a string
function sample(image_path::String)
    img = Metalhead.preprocess(load(image_path)) |> gpu
    features = vgg(img) |> cpu
    
    reset(rnn_decoder)
    prev_word = "<s>"
    lstm_inp = cnn_encoder(features)
    lstm_out = rnn_decoder(lstm_inp)
    output = ""
    
    for i in 1:15
        output = string(output," ",prev_word)
        if prev_word == "</s>"
            break
        end
        word_embeddings = embedding(onehotword(prev_word))
        predictions = softmax(decoder(rnn_decoder(word_embeddings)))
        next_word = idx2word[Flux.argmax(predictions)[1]]
        prev_word = next_word
    end
    
    output
end
