## Multi-head attention (GPT)

# GPT is built of a multi-head attention architecture.  We offer here a very small instance based on
# Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).  The default parameters give a
# model much smaller than nanoGPT, tuned for fastest convergence on a very small data set
# (Shakespeare).

# This model takes as input a sequence of existing text (context) and produces as output the
# predicted next character.  Actually, it produces the predicted next character for each initial
# sub-sequence of the input, in effect giving an extra degree of parallelism for the purposes of
# training.

# For the attention mechanism, we use [Flux.MultiHeadAttention](https://fluxml.ai/Flux.jl/stable/reference/models/layers/#MultiHeadAttention).

# To run this example, we need the following packages:

using JLD2
using CUDA, cuDNN
using Flux
using MLUtils
using Random
using Statistics
using StatsBase
using ProgressMeter

device = Flux.get_device()

# With these options, each epoch takes 22 seconds on an RTX 4090.
# Loss is 1.81 after 1 epoch, and generates recognizable text.
# Loss is 1.58 after 5 epochs.
# Loss is 1.52 after 20 epochs.
Base.@kwdef mutable struct Args
    n_embed::Int = 64          # Length of latent vector
    n_hidden::Int = 256        # Hidden dim for MLP layer
    n_heads::Int = 4           # Number of attention heads
    qk_dim::Int = 16           # Attn query/key size, typically n_embed / n_heads
    v_dim::Int = 16            # Attn value size, typically n_embed / n_heads
    n_layers::Int = 6          # Number of attention/MLP layers
    seqlen::Int = 64           # Context length
    batchsz::Int = 128         # Number of sequences in each batch
    dropout::Float32 = 0.0     # Dropout fraction during training
    testpercent::Float64 = 0.1 # Percent of corpus examples to use for testing
    lr::Float64 = 1e-2         # Learning rate
    epochs::Int = 20           # Number of epochs
end



# One layer of the GPT model.  We will have args.n_layers of these.
struct GPTBlock
    layernorm1::LayerNorm
    mha::MultiHeadAttention
    mlp::Chain
end

Flux.@layer GPTBlock

function GPTBlock(; n_embed, n_hidden, qk_dim, v_dim, n_heads, dropout)
    GPTBlock(
        LayerNorm(n_embed),
        MultiHeadAttention(n_embed => (qk_dim, v_dim) => n_embed; nheads=n_heads, dropout_prob=dropout),
        Chain(
            LayerNorm(n_embed),
            Dense(n_embed => n_hidden, gelu),
            Dense(n_hidden => n_embed),
            Dropout(dropout)
        ),
    )
end

function (m::GPTBlock)(x)
    y, α = m.mha(m.layernorm1(x); mask=NNlib.make_causal_mask(x))
    x += y
    x += m.mlp(x)
    return x
end



struct GPT
    alphabet::Vector{Char}
    tok_embed::Embedding
    pos_embed::Embedding
    dropout::Dropout
    blocks::Vector{GPTBlock}
    layernorm1::LayerNorm
    output_layer::Dense
end

Flux.@layer GPT

function GPT(args::Args, alphabet::AbstractVector{Char})
    n_vocab = length(alphabet)
    GPT(
        alphabet,
        Embedding(n_vocab => args.n_embed),
        Embedding(args.seqlen => args.n_embed),
        Dropout(args.dropout),
        map(_ -> GPTBlock(
            n_embed  = args.n_embed,
            n_hidden = args.n_hidden,
            qk_dim   = args.qk_dim,
            v_dim    = args.v_dim,
            n_heads  = args.n_heads,
            dropout  = args.dropout), 1:args.n_layers),
        LayerNorm(args.n_embed),
        Dense(args.n_embed => n_vocab),
    )
end

function (m::GPT)(tokens)
    T, B = size(tokens)
    te = m.tok_embed(tokens)
    pe = m.pos_embed(1:T)
    x = m.dropout(te .+ pe)
    for blk in m.blocks
        x = blk(x)
    end
    x = m.layernorm1(x)
    x = m.output_layer(x)
    return x
end

# Infer args.seqlen from the given model.
context_length(m::GPT) = size(m.pos_embed.weight, 2)



# Use the model to generate some text.
function generate(model, seed, outlen)
    seqlen = context_length(model)
    if isempty(seed)
        seed = "_"
    end
    x = map(c -> findfirst(==(c), model.alphabet)::Int64, collect(seed))
    while length(x) < outlen
        tail = x[max(1, end-seqlen+1):end]
        tail = reshape(tail, length(tail), 1)
        y = model(tail |> device) |> cpu
        p = softmax(y[:,end,1])
        j = sample(1:length(model.alphabet), Weights(p))
        #j = argmax(p)
        #x = vcat(x, [j])
        push!(x, j)
    end
    String(map(j -> model.alphabet[j], x))
end



# Load data from input file, and partition into training and testing subsets.
function getdata(args::Args)
    isfile("input.txt") || download(
        "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
        #"https://cs.stanford.edu/people/karpathy/char-rnn/warpeace_input.txt",
        "input.txt",
    )

    text = String(read("input.txt"))

    # For aesthetic reasons, replace newlines with strings.  This is not necessary, but makes
    # strings print nicer.
    text = replace(text, r"\r?\n" => " ")

    ## an array of all unique characters
    alphabet = [unique(text)..., '_']
    stop = alphabet[end]

    B = (length(text)-1) ÷ args.seqlen
    # We must collect() before indexing, because String indexing does strange things with multi-byte
    # characters and we could end up with the wrong length.
    Xs = reshape(collect(text)[1:B*args.seqlen], args.seqlen, B)
    Ys = reshape(collect(text)[2:B*args.seqlen+1], args.seqlen, B)

    # Input string starts with stop character '_', representing zero context.
    Xs[1,:] .= stop

    # Xs (input) should consist of indices into `alphabet` because this is what Embedding expects.
    # Ys (output) should be one-hot because this is what logitcrossentropy expects.
    Xs = map(c -> Int32(findfirst(==(c), alphabet)), Xs)
    Ys = Flux.onehotbatch(Ys, alphabet)
    #@show Xs |> typeof # = Matrix{Int32}
    #@show Xs |> size   # = (64, 71458)
    #@show Ys |> typeof # = OneHotArrays.OneHotArray{UInt32, 2, 3, Matrix{UInt32}}
    #@show Ys |> size   # = (68, 64, 71458)

    numbatch = size(Xs, 2)
    split = floor(Int, (1-args.testpercent) * numbatch)

    trainX, trainY = Xs[:,1:split],       Ys[:,:,1:split]
    testX,  testY =  Xs[:,(split+1):end], Ys[:,:,(split+1):end]

    return (alphabet, trainX, trainY, testX, testY)
end



function train(; kws...)
    args = Args(; kws...)

    @info "Training on $device"

    # Load data from input file, and partition into training and testing subsets.
    alphabet, trainX, trainY, testX, testY = getdata(args)

    # Move data to the device (CPU or GPU).
    trainX, trainY, testX, testY = device.((trainX, trainY, testX, testY))

    @info "Alphabet size: $(length(alphabet))"
    @info "Training size: $(size(trainX, 2)) sequences."
    @info "Testing  size: $(size(testX,  2)) sequences."

    # This will iterate over the training data, giving us batches of size args.batchsz.
    loader = MLUtils.DataLoader((trainX, trainY), batchsize=args.batchsz, shuffle=true)

    # Construct the model.
    model = GPT(args, alphabet) |> device
    @info "Number of params: $(sum(length, Flux.params(model)))"

    function loss(m, xs, ys)
        return sum(Flux.logitcrossentropy(m(xs), ys))
    end

    opt_state = Flux.setup(Adam(args.lr), model)
    #opt_state = JLD2.load("model-checkpoint.jld2", "opt_state")

    for epoch = 1:args.epochs
        @info "Training, epoch $(epoch) / $(args.epochs)"
        trainmode!(model) # Enable dropout, for training
        @showprogress for (x,y) in loader
            grad = Flux.gradient(model) do m
                loss(m, x, y)
            end
            Flux.update!(opt_state, model, grad[1])
        end

        testmode!(model) # Disable dropout, for testing/inference

        # Save model checkpoint.
        jldsave("model-checkpoint.jld2",
            model_state=Flux.state(model |> cpu),
            opt_state=opt_state,
            alphabet=alphabet,
            args=args)

        # Show loss per character for the testing dataset.
        @show loss(model, testX, testY)

        # Generate some text.  The character "_" is the stop character, and we're using it here to
        # represent that we are starting with zero context.
        for _ in 1:5
            @show generate(model, "_", 50)
        end
    end

    return args, model
end

# Load a model from a checkpoint (see `jldsave` above).
function load_model(filename)
    args = JLD2.load(filename, "args")
    alphabet = JLD2.load(filename, "alphabet")
    model = GPT(args, alphabet)
    model_state = JLD2.load(filename, "model_state")
    model = Flux.loadmodel!(model, model_state);
    return args, model
end

if true
    args, model = train()
else
    args, model = load_model("model-checkpoint.jld2") |> device
end

generate(model, "The", 50)
