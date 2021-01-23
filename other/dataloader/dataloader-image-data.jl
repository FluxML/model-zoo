using MLDatasets: MNIST
using Flux.Data: DataLoader
using Flux: onehotbatch
using Parameters: @with_kw

@with_kw mutable struct Args
    minibath_size::Int = 128  # Size of mini-batch
end

function get_data(args)

    # Load the MNIST train and test data from MLDatasets
    train_x, train_y = MNIST.traindata(Float32)
    test_x, test_y = MNIST.testdata(Float32)

    # Reshape data to 28x28x1 multi-dimensional array
    train_x = reshape(train_x, 28, 28, 1, :)
    test_x = reshape(test_x, 28, 28, 1, :)

    # Labels must be encoded as a vector with the same dimension 
    # as the number of categories (unique handwritten digits) in the data set
    train_y, test_y = onehotbatch(train_y, 0:9), onehotbatch(test_y, 0:9)

    # Now, we load the train and test images and their labels onto a DataLoader object
    data_loader_train = DataLoader(train_x, train_y, batchsize=args.minibath_size, shuffle=true)
    data_loader_test = DataLoader(train_x, train_y, batchsize=args.minibath_size, shuffle=true)

    return data_loader_train, data_loader_test

end

function train(; kws...)
    args = Args(; kws...)

    @info("Loading data...")
    data_loader_train, data_loader_test = get_data(args)

    # Iterating over train data
    for (x, y) in data_loader_train
        @assert size(x) == (28, 28, 1, 128) || size(x) == (28, 28, 1, 96)
        @assert size(y) == (10, 128) || size(y) == (10, 96)
     end

end

cd(@__DIR__)
train()

