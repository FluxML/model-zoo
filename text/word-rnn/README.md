# Word-level generative language model

Created using julia 1.7.2

## Introduction
This model and training sequence is based on the model in https://arxiv.org/pdf/1409.2329.pdf, and uses LSTMs with
dropout layers to avoid large RNNs tendency to over fit.

## Model Architecture & Parameters

Customizable Parameters
```julia
# Model Params
em_size::Int = 200
recur_size::Int = 400
clip::Float64 = 0.1
dropout::Float64 = 0.25

# Training Params
γ::Float64 = 2.5
epochs::Int = 8

# Data Params
nbatch::Int = 250
seqlen::Int = 20
```

Model Architecture
```julia
## word-rnn.jl
Flux.Embedding(vocab_size, args.em_size),
Flux.Dropout(args.dropout),
Flux.LSTM(args.em_size, args.recur_size),
Flux.Dropout(args.dropout),
Flux.LSTM(args.recur_size, args.recur_size),
Flux.Dropout(args.dropout),
Flux.Dense(args.recur_size, vocab_size)
```

## Corpus
The model is trained on Plato's republic, freely available from MIT 
[here](http://classics.mit.edu/Plato/republic.mb.txt). The corpus is parsed in [corpus.jl](corpus.jl)

## Training and Evaluation

Set up the word-level generative language model project and install 
dependencies from the `text/word-rnn` directory:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Then begin training the model. This will take 20-25 minutes on a Macbook M1 chip, and 2-4 minutes 
on an NVIDIA GPU with CUDA installed.
```julia
include("word-rnn.jl")
```

The model's final validation perplexity should be approximately `180 bits`, which is comparable to the model performance 
in the paper accounting for the differences in the corpus and training time.

Once the model's training is complete, the `word-rnn.jl` script will automatically sample from the model to 
produce three random text sequences of length 20 using the following seeds, `socrates` and `liberty`, as well as 
using a randomly sampled seed from the corpus. Example output:

```shell
socrates should have been such said as reason they were them selected and take his UNK will binding not reduced me

liberty and fain them in knowledge there were guilty many subjects which a tempers the introduction numerous like to alas yes

vulgarity received and behold amusements and really and they UNK their qualities which were commanded at their own own UNK mind
```

To manually produce text sequences from the model use the `sample` method provided in `word-rnn.jl`. For example,
```julia
# 50 word sequence with random seed
sample(model, vocab, word2ind, 50) |> println

# 25 word sequence with "freedom" as the seed
sample(model, vocab, word2ind, 25; seed="freedom") |> println
```

