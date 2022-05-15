# architecture based on https://arxiv.org/pdf/1409.2329.pdf
include("corpus.jl")
using Flux
using Parameters: @with_kw
using Statistics: mean
using StatsBase: wsample

@with_kw mutable struct Args
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
end

function create_model(vocab_size, args)
    return Chain(
        Flux.Embedding(vocab_size, args.em_size),
        Flux.Dropout(args.dropout),
        Flux.LSTM(args.em_size, args.recur_size),
        Flux.Dropout(args.dropout),
        Flux.LSTM(args.recur_size, args.recur_size),
        Flux.Dropout(args.dropout),
        Flux.Dense(args.recur_size, vocab_size)
    )
end

function train(; kws...)
    # initialize parameter struct
    args = Args()

    # load train data and vocabulary
    x_train, y_train, word2ind, vocab = get_data(args)
    vocab_size = length(vocab)

    # create model
    model = create_model(vocab_size, args; kws...)

    # logit cross entropy loss function
    function loss(x, y)
        Flux.reset!(model)
        return mean(Flux.logitcrossentropy(model(x_i), y_i) for (x_i, y_i) in zip(x,y))
    end

    # reference to model params, and optimizer
    ps = Flux.params(model)

    # create batch iterators for data and validation
    data_loader = zip(x_train[1:end-5], y_train[1:end-5])
    hold_out = zip(x_train[end-5:end], y_train[end-5:end])

    # used for updating hyperparameters
    best_val_loss = nothing
    lr = args.γ

    # begin training loop
    @info "Start Training, total $(args.epochs) epochs"
    for epoch = 1:args.epochs

        @info "Epoch $(epoch) / $(args.epochs)"

        for batch in data_loader

            gradient = Flux.gradient(ps) do
                # compute loss for this batch
                training_loss = loss(batch...)
                return training_loss
            end

            for x in ps
                # apply clip to handle exploding gradients
                grad_x = clamp!(gradient[x], -args.clip, args.clip)
                # backprop
                x .-= lr .* grad_x
            end
        end

        # compute and show the loss for the hold out set
        validation_loss = mean([loss(x_v, y_v) for (x_v, y_v) in hold_out])
        @show(validation_loss, lr)

        if best_val_loss == nothing || validation_loss < best_val_loss
            best_val_loss = validation_loss
        else
            # Anneal the learning rate if hold out set loss did not improve
            lr /= 4.0
        end
    end

    #= For a more simple training loop:
    # training callback for loss on hold-out set
    evalcb() = @show(mean([loss(x_v, y_v) for (x_v, y_v) in hold_out]))
    throttled_cb = Flux.throttle(evalcb, 5)
    opt = ADAM(args.γ)
    @Flux.epochs args.epochs Flux.train!(loss, ps, data_loader, opt, cb=throttled_cb)
    =#

    # show final lr, and hold out set perplexity
    valid_perplex = exp(mean([loss(x_v, y_v) for (x_v, y_v) in hold_out]))
    @info "Training finished, final validation perplexity: $(valid_perplex) bits, final lr: $(lr)"

    return model, vocab, word2ind
end

function sample(model, vocab, word2ind, len; seed="")
    # load the model, and generate a sentence of length `len`
    model = cpu(model)
    Flux.reset!(model)
    buf = IOBuffer()
    if seed == ""
        seed = string(rand(vocab))
    end
    write(buf, seed)
    c = wsample(vocab, Flux.softmax(model(word2ind[seed])))
    for i = 1:len
        write(buf, ' ')
        write(buf, c)
        c = wsample(vocab, softmax(model(word2ind[c])))
    end
    write(buf, '\n')
    return String(take!(buf))
end

cd(@__DIR__)
@time begin
    model, vocab, word2ind = train()
end
@info "Word language model generation examples:"
sample(model, vocab, word2ind, 20; seed="socrates") |> println
sample(model, vocab, word2ind, 20; seed="liberty") |> println
sample(model, vocab, word2ind, 20) |> println
