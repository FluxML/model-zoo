using Flux
using Random
using Statistics
include("utils.jl")

# Hyperparameters and configuration of AR process
@Base.kwdef mutable struct Args
    seed::Int            = 72                  # Random seed
    # AR process parameters
    ϕ::Vector{Float32}   = [.3f0, .2f0, -.5f0] # AR coefficients (=> AR(3))
    proclen::Int         = 750                 # Process length
    # Recurrent net parameters
    dev                  = cpu                 # Device: cpu or gpu
    opt                  = ADAM                # Optimizer
    η::Float64           = 2e-3                # Learning rate
    hidden_nodes::Int    = 64                  # Number of hidden nodes
    hidden_layers::Int   = 2                   # Number of hidden layers
    layer                = LSTM                # Type of layer, should be one of LSTM, GRU, RNN
    epochs::Int          = 100                 # Number of epochs
    seqlen::Int          = 10                  # Sequence length to use as input
    seqshift::Int        = 10                  # Shift between sequences (see utils.jl)
    train_ratio::Float64 = .7                  # Percentage of data in the train set
    verbose::Bool        = true                # Whether we log the results during training or not
end

# Creates a model according to the pre-defined hyperparameters `args`
function build_model(args)
    Chain(
        args.layer(1, args.hidden_nodes),
        [args.layer(args.hidden_nodes, args.hidden_nodes) for _ ∈ 1:args.hidden_layers-1]...,
        Dense(args.hidden_nodes, 1, identity)
    ) |> args.dev
end

# Creates training and testing samples according to hyperparameters `args`
function generate_train_test_data(args)
    # Generate full AR process
    data = generate_process(args.ϕ, args.proclen)
    # Create input X and output y (series shifted by 1)
    X, y = data[1:end-1], data[2:end]
    # Split data into training and testing sets
    idx = round(Int, args.train_ratio * length(X))
    Xtrain, Xtest = X[1:idx], X[idx+1:end]
    ytrain, ytest = y[1:idx], y[idx+1:end]
    # Transform data to time series batches and return
    map(x -> batch_timeseries(x, args.seqlen, args.seqshift) |> args.dev, 
        (Xtrain, Xtest, ytrain, ytest))
end

function mse_loss(model, x, y)
    # Warm up recurrent model on first observation
    model(x[1])
    # Compute mean squared error loss on the rest of the sequence
    mean(Flux.Losses.mse.([model(xᵢ) for xᵢ ∈ x[2:end]], y[2:end]))
end

# Trains and outputs the model according to the chosen hyperparameters `args`
function train_model(args)
    Random.seed!(args.seed)
    # Create recurrent model
    model = build_model(args)
    # Get data
    Xtrain, Xtest, ytrain, ytest = generate_train_test_data(args)

    opt = Flux.setup(args.opt(args.η), model)
    # Training loop
    for i ∈ 1:args.epochs
        Flux.reset!(model) # Reset hidden state of the recurrent model
        # Compute the gradients of the loss function
        (∇m,) = gradient(model) do m
            mse_loss(m, Xtrain, ytrain)
        end
        Flux.update!(opt, model, ∇m) # Update model parameters
        if args.verbose && i % 10 == 0 # Log results every 10 epochs
            # Compute loss on train and test set for logging (important: the model must be reset!)
            Flux.reset!(model)
            train_loss = mse_loss(model, Xtrain, ytrain)
            Flux.reset!(model)
            test_loss = mse_loss(model, Xtest, ytest)
            @info "Epoch $i / $(args.epochs), train loss: $(round(train_loss, digits=3)) | test loss: $(round(test_loss, digits=3))"
        end
    end
    return model
end

cd(@__DIR__)

args = Args()    # Set up hyperparameters
m = train_model(args)   # Train and output model
