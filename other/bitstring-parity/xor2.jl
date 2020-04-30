include("data.jl")
using Flux, Statistics
using Flux: onehot, onehotbatch, throttle, logitcrossentropy, reset!, onecold
using Parameters: @with_kw

@with_kw mutable struct Args
    lr::Float64 = 1e-3    # Learning rate
    epochs::Int = 20      # Number of epochs for training
    train_len::Int = 2000  # Length of training data to be generated
    val_len::Int = 100     # Length of Validation Data
    throttle::Int = 10    # Throttle timeout
end

function getdata(args)
    # training data of bit strings from length 2 to 10
    train = gendata(args.train_len, 1:10)
    # validation data of bit strings of length 10
    val = gendata(args.val_len, 10)
    return train, val
end

function build_model()
    scanner = LSTM(length(alphabet), 20)
    encoder = Dense(20, length(alphabet))
    return scanner, encoder
end

function model(x, scanner, encoder)
    state = scanner.(x.data)[end]
    reset!(scanner)
    encoder(state)
end

function train(; kws...)
    # Initialize the parameters
    args = Args(; kws...)
    
    # Load Data 
    train_data, val_data = getdata(args)

    @info("Constructing Model...")
    scanner,encoder = build_model()
   
    loss(x, y) = logitcrossentropy(model(x, scanner, encoder), y)
    batch_loss(data) = mean(loss(d...) for d in data)

    opt = ADAM(args.lr)
    ps = params(scanner, encoder)
    evalcb = () -> @show batch_loss(val_data)

    @info("Training...")
    for i=1:args.epochs
        Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, args.throttle))
    end

    # Try running the model on strings of length 50.
    #
    # Even though the model has only been trained with
    # much shorter strings, it has learned the
    # parity function and will accurate on longer strings.
    function t50()
        l = batch_loss(gendata(1000, 50))
        println("Batch_loss for length 50 string: ", l,"\n")
    end
    t50()
    return scanner, encoder
end

function test(scanner, encoder)
    # sanity test
    tx = map(c -> onehotbatch(c, alphabet), [
        [false, true], # 01 -> 1
        [true, false], # 10 -> 1
        [false, false], # 00 -> 0
        [true, true]]) # 11 -> 0
    @info("Test...")
    out = [onecold(model(x, scanner, encoder)) - 1 for x in tx]
    input = [[0,1],[1,0],[0,0],[1,1]]
    for i in 1:length(tx)
        print(input[i]," => ",out[i],"\n")
    end	
end

cd(@__DIR__)
scanner, encoder = train()
test(scanner, encoder)
