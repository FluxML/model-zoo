# # Seq2seq translation in Flux
# In this notebook, I share the code I wrote to make a seq2seq nmt-model (neural machine translation model) to translate simple english sentences to french.
# The code is written in Julia using Flux.
#
# ### note:
# For some reason, when I train the model for some epochs, I get gibberish results.
#
# |Input (Eng)|Prediction (*Fr*)|Expected (Fr)|
# | - | - | - |
# |"You are too skinny"|"Vous êtes ' que . . . . . . . ."| "Vous êtes trop maigre"  |
# |"He is painting a picture"|"Il est est de . . . . . . . ."|"Il est en train de peindre un tableau"|
# | ... | ... | ... |
# If you know what I'm doing wrong, please do let me know!

# # The data
# The english-french sentence pairs dataset is found on this website: http://www.manythings.org/anki/fra-eng.zip.
# For the data preparation, I mainly follow the official Pytorch tutorial on seq2seq models: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html.
#
# We create a `Lang` struct which holds two dictionaries to convert words to indices and back. Every `Lang` instance gets instantiated with a SOS-(start of sentence), EOS(end of sentence)-, UNK(unknown word) and PAD(padding)-token.
# Padding is necessary because we will be training in batches of differently sized sentences.
#
#
# *Since the data is relatively small (a lot of sentences get filtered out), we keep all words instead of discarding scarcely used words.
# This means the `UNK` token will not be used.*
#
# The function `readLangs` takes the text file, splits up the sentences (which are tab-delimited) and intantiates two new languages (lang1 and lang2). and assigns them to two newly created languages.

using CuArrays, Flux, Statistics, Random

FILE = "D:/downloads/fra-eng/eng-fra.txt"

mutable struct Lang
    name
    word2index
    word2count
    index2word
    n_words
end

Lang(name) = Lang(
    name,
    Dict{String, Int}(),
    Dict{String, Int}(),
    Dict{Int, String}(1=>"SOS", 2=>"EOS", 3=>"UNK", 4=>"PAD"),
    4)

function (l::Lang)(sentence::String)
    for word in split(sentence, " ")
            if word ∉ keys(l.word2index)
                l.word2index[word] = l.n_words + 1
                l.word2count[word] = 1
                l.index2word[l.n_words + 1] = word
                l.n_words += 1
            else
                l.word2count[word] += 1
            end
    end
end

function normalizeString(s)
    s = strip(lowercase(s))
    s = replace(s, r"([.!?,])"=>s" \1")
    s = replace(s, "'"=>" ' ")
    return s
end

function readLangs(lang1, lang2; rev=false)
    println("Reading lines...")
    lines = readlines(FILE)
    pairs = [normalizeString.(pair) for pair in split.(lines, "\t")]
    if rev
        pairs = reverse.(pairs)
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    end
    return(input_lang, output_lang, pairs)
end

# As suggested in the Pytorch tutorial, we create a function to filter out sentences that don't start with `english_prefixes` ("i am", "i'm", "you are"...), as well as sentences that exceed the `MAX_LENGTH` (which is set to 10).
#
# The function `prepareData` takes the names of two languages and creates these language instances as well as the sentence pairs by calling `readLangs`.
# After the sentence pairs get filtered (with `filterPair`), every unique word in a sentence get's added to the corresponding language's vocabulary (`word2index`, `index2word`, `n_words`) while every additional instance of a word increments `n_words` by 1.
#
# Sentences from the input language are added to `xs`, target sentences are added to `ys`. Finally, inputs (`xs`) and targets (`ys`) are shuffled.

MAX_LENGTH = 10

eng_prefixes = [
    "i am ", "i ' m ",
    "he is ", "he ' s ",
    "she is ", "she ' s ",
    "you are ", "you ' re ",
    "we are ", "we ' re ",
    "they are ", "they ' re "]

function filterPair(pair)
    return(false ∉ (length.(split.(pair, " ")) .<= MAX_LENGTH) && true ∈ (startswith.(pair[1], eng_prefixes)))
end

function prepareData(lang1, lang2; rev=false)
    input_lang, output_lang, pairs = readLangs(lang1, lang2; rev=rev)
    println("Read $(length(pairs)) sentence pairs.")
    pairs = [pair for pair in pairs if filterPair(pair)]
    println("Trimmed to $(length(pairs)) sentence pairs.\n")
    xs = []
    ys = []
    for pair in pairs
        push!(xs, pair[1])
        push!(ys, pair[2])
    end
    println("Counting words...")
    for pair in pairs
        input_lang(pair[2])
        output_lang(pair[1])
    end
    println("Counted words:")
    println("• ", input_lang.name, ": ", input_lang.n_words)
    println("• ", output_lang.name, ": ", output_lang.n_words)
    return(input_lang, output_lang, xs, ys)
end

fr, eng, xs, ys = prepareData("fr", "eng")
indices = shuffle([1:length(xs)...])
xs = xs[indices]
ys = ys[indices];

# The function `indexesFromSentence` takes a language's `word2index` and maps all the words in a sentence to a index, later this index will get used to get the word's embedding. Note that, at the end of every sentence, the `EOS`-index (2) gets added, this is for the model to know when to stop predicting during inference.
#
# To make batches for mini-batch training, the data (`[indexesFromSentence.([eng], xs), indexesFromSentence.([fr], ys)]`) gets split in chunks of `BATCH_SIZE`. Since sentences in a chunk often have different lengths, the `PAD`-index (4), gets added to the end of sentences to make them as long as the longest sentence of the chunk.
#
# To be able to easily pass a chunk to an RNN, the n<sup>th</sup> word of every sentence in the chunk get placed next to each other in an array. Also, all the words get OneHot encoded.
#
# ![batching](D:/downloads/background.svg)

BATCH_SIZE = 32

indexesFromSentence(lang, sentence) = append!(get.(Ref(lang.word2index), split(lowercase(sentence), " "), 3), 2)

function batch(data, batch_size, voc_size; gpu=true)
    chunks = Iterators.partition(data, batch_size)
    batches = []
    for chunk in chunks
        max_length = maximum(length.(chunk))
        chunk = map(sentence->append!(sentence, fill(4, max_length-length(sentence))), chunk)
        chunk = hcat(reshape.(chunk, :, 1)...)
        batch = []
        for i in 1:size(chunk, 1)
            if gpu
                push!(batch, cu(Flux.onehotbatch(chunk[i, :], [1:voc_size...])))
            else
                push!(batch, Flux.onehotbatch(chunk[i, :], [1:voc_size...]))
            end
        end
        push!(batches, batch)
    end
    return(batches)
end

x, y = batch.([indexesFromSentence.([eng], xs), indexesFromSentence.([fr], ys)], [BATCH_SIZE], [eng.n_words, fr.n_words]; gpu=true);

# # The Model
#
# For the model, we're using a **encoder-decoder** architecture.
# ![encoder-decoder](https://smerity.com/media/images/articles/2016/gnmt_arch_attn.svg)
# *image source: https://smerity.com/articles/2016/google_nmt_arch.html*
#
# ### High level overview
# The **encoder** takes the OneHot-encoded words and uses the embedding layer to get their embedding, a multidimensional-representation of that word. Next, the words get passed through a RNN (in our case a GRU). For each word, the RNN spits out a state-vector (encoder-outputs).
#
# The job of the **decoder** is to take the output of the encoder and mold it into a correct translation of the original sentence. The **attention** layer acts as a guide for the decoder. Every timestep (every time the decoder is to predict a word), it takes all the encoder-outputs and creates **one** state vector (the context vector) with the most relevant information for that particular timestep.

## some constants to be used for the model:
EMB_size = 128
HIDDEN = 128
LEARNING_RATE = 0.005
DROPOUT = 0.2;

# For the encoder, we're using a bidirectional GRU, the input is read from front to back as well as from back to front. This should help for a more robust `encoder_output`.
# The `Flux.@treelike` macro makes sure all the parameters are recognized by the optimizer to optimise the values.

struct Encoder
    embedding
    linear
    rnn
    out
end
Encoder(voc_size::Integer; h_size::Integer=HIDDEN) = Encoder(
    param(Flux.glorot_uniform(EMB_size, voc_size)),
    Dense(EMB_size, HIDDEN, relu),
    GRU(h_size, h_size),
    Dense(h_size, h_size))
function (e::Encoder)(x; dropout=0)
    x = map(x->Dropout(dropout)(e.embedding*x), x)
    x = e.linear.(x)
    enc_outputs = e.rnn.(x)
    h = e.out(enc_outputs[end])
    return(enc_outputs, h)
end
Flux.@treelike Encoder

# The decoder takes the word it predicted in the previous timestep as well the `encoder_outputs`. The context vector gets created by passing these `encoder_outputs` as well as the current state of the decoder's RNN to the attention layer. Finally, the context vector is concatenated with the word of the previous timestep to predict the word of the current timestep.
#
# *During the first timestep, the decoder doesn't have acces to a previously predicted word. To combat this, a `SOS`-token is provided*

struct Decoder
    embedding
    linear
    attention
    rnn
    output
end
Decoder(h_size, voc_size) = Decoder(
    param(Flux.glorot_uniform(EMB_size, voc_size)),
    Dense(EMB_size, HIDDEN),
    Attention(h_size),
    GRU(h_size*2, h_size),
    Dense(h_size, voc_size, relu))
function (d::Decoder)(x, enc_outputs; dropout=0)
    x = d.embedding * x
    x = Dropout(dropout)(x)
    x = d.linear(x)
    decoder_state = d.rnn.state
    context = d.attention(enc_outputs, decoder_state)
    x = d.rnn([x; context])
    x = softmax(d.output(x))
    return(x)
end
Flux.@treelike Decoder

# For the attention mechanism, we follow the implementation from the paper "Grammar as a Foreign Language" (https://arxiv.org/pdf/1412.7449.pdf).
#
# Esentially, the encoder outputs and the hidden state of the decoder are used to a context vector which contains all the necessary information to decode into a translation during a particular timestep.
# The paper shows the following equations:
#
# $ u_i^t = v^T tanh(W_1'h_i+W_2'd_t) $
#
# $ a_i^t = softmax(u_i^t) $
#
# $ \sum\limits_{i=1}^{T_a} a_i^t h_i$
#
# Where the encoder hidden states are denoted `(h1, . . . , hTA )` and we denote the hidden states of the decoder by `(d1, . . . , dTB )`

struct Attention
    W1
    W2
    v
end
Attention(h_size) = Attention(
    Dense(h_size, h_size),
    Dense(h_size, h_size),
    param(Flux.glorot_uniform(1, h_size)))
function (a::Attention)(enc_outputs, d)
    U = [a.v*tanh.(x) for x in a.W1.(enc_outputs).+[a.W2(d)]]
    A = softmax(vcat(U...))
    out = sum([gpu(collect(A[i, :]')) .* h for (i, h) in enumerate(enc_outputs)])
end
Flux.@treelike Attention

testEncoder = Encoder(eng.n_words)|>gpu
testDecoder = Decoder(HIDDEN, fr.n_words)|>gpu;

# The model function is made to return the loss when the input and the target are provided.
# The hidden states of the RNN from both the encoder as well as the decoder are reset, by doing this you make sure no information of previous sentences is remembered.
#
# The encoder_ouputs are made by passing the input through the encoder, the initial decoder input is made and the decoder's rnn state is initialized with the last encoder output.
# The decoder has to predict `max_length` words with `max_length` being the length of the longes sentence.
#
# First off, the model decides whether teacher forcing will be used this timestep. Teacher forcing means instead of using the decoder output as the next timestep's decoder input, the correct input is used. Teacher forcing is especially useful in the beginning of training since decoder outputs won't make sense.
#
# Every timestep, the decoder's prediction as well as the correct target are passed to a loss function. All the losses of all timesteps are summed up and returned.

function model(encoder::Encoder, decoder::Decoder, x, y; teacher_forcing = 0.5, dropout=DROPOUT, voc_size=fr.n_words)
    total_loss = 0
    max_length = length(y)
    batch_size = size(x[1], 2)
    Flux.reset!.([encoder, decoder])
    enc_outputs, h = encoder(x; dropout=dropout)
    decoder_input = Flux.onehotbatch(ones(batch_size), [1:voc_size...])
    decoder.rnn.state = h
    for i in 1:max_length
        use_teacher_forcing = rand() < teacher_forcing
        decoder_output = decoder(decoder_input, enc_outputs; dropout=dropout)
        total_loss += loss(decoder_output, y[i])
        if use_teacher_forcing
            decoder_input = y[i]
        else
            decoder_input = Flux.onehotbatch(Flux.onecold(decoder_output.data), [1:voc_size...])
        end
    end
    return(total_loss)
end

model(x, y) = model(testEncoder, testDecoder, x, y; dropout = DROPOUT)

# When the target is not provided to the `model` function, the model returns a prediction instead of a loss value.
#
#
# *Note that, when the model is trained, the loop could be set to run indefinitely because the loop will break when an `EOS`-token is predicted.
# I've set the loop to run for an arbitrary amount of timesteps (in this case 12) because the model doesn't seem to be able to learn to predict an `EOS token`*

function model(encoder::Encoder, decoder::Decoder, x; reset=true, voc_size=fr.n_words)
    result = []
    if reset Flux.reset!.([encoder, decoder]) end
    enc_outputs, h = encoder(x)
    decoder_input = Flux.onehot(1, [1:voc_size...])
    decoder.rnn.state = h
    for i in 1:12
        decoder_output = Flux.onecold(decoder(decoder_input, enc_outputs))
        if decoder_output[1] == 2 break end
        push!(result, decoder_output...)
    end
    return(result)
end

# The `loss` function expects a probability distribution over all possible words in the vocabulary, this gets accounted for by the softmax layer in the decoder. The loss function itself is crossentropy (a.k.a. negative-log-likelihood).
# We pass an vector of ones, except for the `PAD`-index (4) as weight to the loss function. This way the model will disregard any predictions that should have been PAD, since padding only occurs after the sentence has ended.
#
#
# For the optimizer, we use ADAM.

lossmask = ones(fr.n_words)|>gpu
lossmask[4] = 0

loss(logits, target) = Flux.crossentropy(logits, target; weight=lossmask)

opt = ADAM(LEARNING_RATE)
ps = params(testEncoder, testDecoder)

# The data (`x` and `y`) gets passed to `partitionTrainTest` to split the data in a train and a test set.
#
# Finally the model is trained for a number of epochs. Every epoch, the loss on the test set gets printed.

function partitionTrainTest(x, y, at)
    n = length(x)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    train_x, test_x = x[train_idx,:], x[test_idx,:]
    train_y, test_y = y[train_idx,:], y[test_idx,:]
    return(train_x, train_y, test_x, test_y)
end

train_x, train_y, test_x, test_y = partitionTrainTest(x, y, 0.90);

EPOCHS = 5

for i in 1:EPOCHS
    Flux.train!(model, ps, zip(train_x, train_y), opt)
    println("loss: ", mean(model.(test_x, test_y)).data)
end

# The `predict` function takes an encoder, decoder and an english sentence. It converts the sentence into it's OneHot representation and passes it to the `model` function. The output gets converted back to a string and returned.

function predict(encoder, decoder, sentence::String)
    sentence = normalizeString(sentence)
    input = append!(get.(Ref(eng.word2index), split(lowercase(sentence), " "), 3), 2)
    input = [Flux.onehot(word, [1:eng.n_words...]) for word in input]
    output = model(encoder, decoder, input)
    output = get.(Ref(fr.index2word), output, "UNK")
    println(output)
end

predict(testEncoder, testDecoder, "she's doing her thing")
predict(testEncoder, testDecoder, "you're too skinny")
predict(testEncoder, testDecoder, "He is singing")

# As you can see, when I run the model for 70 epochs, the results are quite underwhelming... Even though sentence subjects are mostly correct, most part of the translation does not make sense.
#
# If you have a look at the loss on the test set during these 70 epochs, you can clearly see the model seems to hit a barrier around 18.
#
# I'm not sure why this is happening and I'd love to know! If you've got an idea on how to improve/fix this model, definitely let me know.
#
# Thanks
# ![encoder-decoder](C:/users/Jules/desktop/grafiek.svg)
