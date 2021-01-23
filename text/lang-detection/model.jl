using Flux
using Flux: onehot, onehotbatch, logitcrossentropy, reset!, throttle
using Statistics: mean
using Random
using Unicode
using Parameters: @with_kw

@with_kw mutable struct Args
    lr::Float64 = 1e-3     # learning rate
    N::Int = 15            # Number of perceptrons in hidden layer
    test_len::Int = 100    # length of test data
    langs_len::Int = 0     # Number of different languages in Corpora
    alphabet_len::Int = 0  # Total number of characters possible, in corpora
    throttle::Int = 10     # throttle timeout
end

function get_processed_data(args)
    corpora = Dict()

    for file in readdir("corpus")
        lang = Symbol(match(r"(.*)\.txt", file).captures[1])
        corpus = split(String(read("corpus/$file")), ".")
        corpus = strip.(Unicode.normalize.(corpus, casefold=true, stripmark=true))
        corpus = filter(!isempty, corpus)
        corpora[lang] = corpus
    end

    langs = collect(keys(corpora))
    args.langs_len = length(langs)
    alphabet = ['a':'z'; '0':'9'; ' '; '\n'; '_']
    args.alphabet_len = length(alphabet)

    # See which chars will be represented as "unknown"
    unique(filter(x -> x ∉ alphabet, join(vcat(values(corpora)...))))

    dataset = [(onehotbatch(s, alphabet, '_'), onehot(l, langs)) for l in langs for s in corpora[l]] |> shuffle

    train, test = dataset[1:end-args.test_len], dataset[end-args.test_len+1:end]
    return train, test
end

function build_model(args)
    scanner = Chain(Dense(args.alphabet_len, args.N, σ), LSTM(args.N, args.N))
    encoder = Dense(args.N, args.langs_len)
    return scanner, encoder
end
 
function model(x, scanner, encoder)
    state = scanner.(x.data)[end]
    reset!(scanner)
    encoder(state)
end

function train(; kws...)
    # Initialize Hyperparameters
    args = Args(; kws...)
    # Load Data
    train_data, test_data = get_processed_data(args)

    @info("Constructing Model...")
    scanner, encoder = build_model(args)

    loss(x, y) = logitcrossentropy(model(x, scanner, encoder), y)
    testloss() = mean(loss(t...) for t in test_data)
    
    opt = ADAM(args.lr)
    ps = params(scanner, encoder)
    evalcb = () -> @show testloss()
    @info("Training...")
    Flux.train!(loss, ps, train_data, opt, cb = throttle(evalcb, args.throttle))
end

cd(@__DIR__)
train()
