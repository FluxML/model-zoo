# # Recursive net on IMDB sentiment treebank

# In this example, we create a recursive neural network to perform sentiment analysis using 
# IMDB data. 
# This type of model can be used 
# for learning tree-like structures (directed acyclic graphs). 
# It computes compositional vector representations for prhases of variable length 
# which are used as features for performing classification. 

# ![treebank](../treebank/docs/treebank.png)

# [Source](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)

# This example uses the [Standford Sentiment Treebank dataset 
# (SST)](https://nlp.stanford.edu/sentiment/index.html) which is often used 
# as one of the benchmark datasets to test new language models. 
# It has five different classes (very negative to very positive) and the 
# goal is to perform sentiment analysis.

# To run this example, we need the following packages:

using Flux
using Flux: logitcrossentropy, throttle
using Flux.Data: Tree, children, isleaf
using Parameters: @with_kw

# The script `data.jl` contains the function `getdata` that obtains
# and process the SST dataset.

include("data.jl")

# We set default values for the hyperparameters:

@with_kw mutable struct Args
    lr::Float64 = 1e-3    ## Learning rate
    N::Int = 300
    throttle::Int = 10    ## Throttle timeout
end

# ## Build the model

# The function `train` loads the data, builds and trains the model. 
# For more information on how the recursive neural network works, see 
# section 4 of [Recursive Deep Models for Semantic Compositionality
# Over a Sentiment Treebank](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf).

function train(; kws...)
    ## Initialize HyperParameters
    args = Args(; kws...)

    ## Load data
    @info("Loading Data...")
    train_data, alphabet = getdata()    

    @info("Constructing model....")
    embedding = randn(Float32, args.N, length(alphabet))

    @info "Size of the embedding" size(embedding)

    W = Dense(2*args.N, args.N, tanh)
    combine(a, b) = W([a; b])

    sentiment = Chain(Dense(args.N, 5))

    function forward(tree)
      if isleaf(tree)
        token, sent = tree.value
        phrase = embedding * token
        phrase, logitcrossentropy(sentiment(phrase), sent)
      else
        _, sent = tree.value
        c1, l1 = forward(tree[1])
        c2, l2 = forward(tree[2])
        phrase = combine(c1, c2)
        phrase, l1 + l2 + logitcrossentropy(sentiment(phrase), sent)
      end
    end

    loss(tree) = forward(tree)[2]
 
    opt = ADAM(args.lr)
    ps = params(embedding, W, sentiment)
    evalcb = () -> @show loss(train_data[1])
    @info("Training Model...")
    Flux.train!(loss, ps, zip(train_data), opt,cb = throttle(evalcb, args.throttle))
end

# ## Train the model

cd(@__DIR__)
train()
