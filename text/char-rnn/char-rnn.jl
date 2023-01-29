# # RNN Character level

# In this example, we create a character-level recurrent neural network. 
# A recurrent neural network (RNN) outputs a prediction and a hidden state at each step 
# of the computation. The hidden state captures historical information of a sequence 
# (i.e., the neural network has memory) and the output is the final prediction of the model.
# We use this type of neural network to model sequences such as text or time series.


# ![char-rnn](../char-rnn/docs/rnn-train.png)

# Source: https://d2l.ai/chapter_recurrent-neural-networks/rnn.html#rnn-based-character-level-language-models

# This example demonstrates the use of Flux’s implementation of the 
# [Long Short Term Memory recurrent layer](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)(LSTM) 
# which is an RNN that generally exhibits a longer memory span over sequences as well as 
# [Flux utility functions](https://fluxml.ai/Flux.jl/stable/utilities/). 

# If you need more information about how RNNs work and related technical concepts, 
# check out the following resources:

# * [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
# * [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
# * [Illustrated Guide to Recurrent Neural Networks: Understanding the Intuition](https://www.youtube.com/watch?v=LHXXI4-IEns)

# To run this example, we need the following packages:

using Flux
using Flux: chunk, batchseq, logitcrossentropy
using OneHotArrays
using StatsBase: wsample
using Base.Iterators: partition
using Random: shuffle

# We set default values for the hyperparameters:

Base.@kwdef mutable struct Args
    lr::Float64 = 1e-2	       # Learning rate
    seqlen::Int = 50	       # Length of batch sequences
    batchsz::Int = 50	       # Number of sequences in each batch
    epochs::Int = 3            # Number of epochs
    usegpu::Bool = false       # Whether or not to use the GPU
    testpercent::Float64 = .05 # Percent of corpus examples to use for testing
end

# ## Data

# We create the function `getdata` to download the training data and create arrays of batches 
# for training the model:


function getdata(args::Args)
    ## Download the data if not downloaded as 'input.txt'
    isfile("input.txt") || download(
        "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
        "input.txt",
    )

    text = String(read("input.txt"))

    ## an array of all unique characters
    alphabet = [unique(text)..., '_']
    stop = '_'

    N = length(alphabet)
    
    ## Partitioning the data as sequence of batches, which are then collected 
    ## as array of batches
    Xs = partition(batchseq(chunk(text, args.batchsz), stop), args.seqlen)
    Ys = partition(batchseq(chunk(text[2:end], args.batchsz), stop), args.seqlen)
    Xs = [Flux.onehotbatch.(bs, (alphabet,)) for bs in Xs]
    Ys = [Flux.onehotbatch.(bs, (alphabet,)) for bs in Ys]

    return Xs, Ys, N, alphabet
end

# The function `getdata` performs the following tasks:

# * Downloads a dataset of [all of Shakespeare's works (concatenated)](https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt) 
# if not previously downloaded. This function loads the data as a vector of characters with the function `collect`.
# * Gets the alphabet. It consists of the unique characters of the data and the stop character ‘_’.
# * One-hot encodes the alphabet and the stop character.
# * Gets the size of the alphabet N.
# * Partitions the data as an array of batches. Note that the `Xs` array contains the sequence of characters in the text whereas the `Ys` array contains the next character of the sequence. 

# ## Model

# We create the RNN with two Flux’s LSTM layers and an output layer of the size of the alphabet:

function build_model(N::Int)
    return Chain(
            LSTM(N => 128),
            LSTM(128 => 128),
            Dense(128 => N))
end 

# The size of the input and output layers is the same as the size of the alphabet. 
# Also, we set the size of the hidden layers to 128. 

# ## Train the model

# Now, we define the function `train` that creates the model and the loss function as well as the training loop:


function train(; kws...)
    ## Initialize the parameters
    args = Args(; kws...)

    ## Select the correct device
    device = args.usegpu ? gpu : cpu
    
    ## Get Data
    Xs, Ys, N, alphabet = getdata(args)

    ## Shuffle and create a train/test split
    L = length(Xs)
    perm = shuffle(1:length(Xs))
    split = floor(Int, (1-args.testpercent) * L)

    trainX, trainY = Xs[perm[1:split]],       Ys[perm[1:split]]
    testX,  testY =  Xs[perm[(split+1):end]], Ys[perm[(split+1):end]]

    ## Move all data to the correct device
    trainX, trainY, testX, testY = device.((trainX, trainY, testX, testY))

    ## Constructing Model
    model = build_model(N) |> device

    function loss(m, xs, ys)
        Flux.reset!(m)
        return sum(logitcrossentropy.([m(x) for x in xs], ys))
    end
    
    ## Training
    opt_state = Flux.setup(Adam(args.lr), model)

    for epoch = 1:args.epochs
        @info "Training, epoch $(epoch) / $(args.epochs)"
        Flux.train!(
            loss,
            model,
            zip(trainX, trainY),
            opt_state
        )
        
        ## Show loss-per-character over the test set
        @show sum(loss.(Ref(model), testX, testY)) / (args.batchsz * args.seqlen * length(testX))
    end
    return model, alphabet
end

# The function `train` performs the following tasks:

# * Calls the function `getdata` to obtain the train and test data as well as the alphabet and its size.
# * Calls the function `build_model` to create the RNN.
# * Defines the loss function. For this type of neural network, we use the [logitcrossentropy](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.logitcrossentropy) 
# loss function. Notice that it is important that we call the function [reset!](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.reset!) 
# before computing the loss so that it resets the hidden state of a recurrent layer back to its original value
# * Sets the [ADAM optimiser](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.RADAM) with the learning rate *lr* we defined above.
# * Creates a [callback](https://fluxml.ai/Flux.jl/stable/training/training/#Callbacks) *evalcb* so that you can observe the training process (print the loss value).
# * Runs the training loop using [Flux’s train!](https://fluxml.ai/Flux.jl/stable/training/training/#Flux.Optimise.train!). 

# ## Test the model

# We define the function `sample_data` to test the model. 
# It generates samples of text with the alphabet that the function `getdata` computed. 
# Notice that it obtains the model’s prediction by calling the 
# [softmax function](https://fluxml.ai/Flux.jl/stable/models/nnlib/#Softmax) 
# to get the probability distribution of the output and then it chooses randomly the prediction.

function sample_data(m, alphabet, len; seed = "")
    m = cpu(m)
    Flux.reset!(m)
    buf = IOBuffer()
    if seed == ""
        seed = string(rand(alphabet))
    end
    write(buf, seed)
    c = wsample(alphabet, softmax([m(onehot(c, alphabet)) for c in collect(seed)][end]))
    for i = 1:len
        write(buf, c)
        c = wsample(alphabet, softmax(m(onehot(c, alphabet))))
    end
    return String(take!(buf))
end

# Finally, to run this example we call the functions `train` and `sample_data`:

cd(@__DIR__)
m, alphabet = train()
sample_data(m, alphabet, 1000) |> println
