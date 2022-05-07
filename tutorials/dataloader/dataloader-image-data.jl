# # An example of DataLoader using image data


# In this example, we show how to load image data in Flux DataLoader and process it in mini-batches. 
# We use the [DataLoader](https://fluxml.ai/Flux.jl/stable/data/dataloader/#Flux.Data.DataLoader) type 
# to handle iteration over mini-batches of data. 
# Moreover, we load the [MNIST dataset](https://juliaml.github.io/MLDatasets.jl/stable/datasets/MNIST/) 
# using the [MLDatasets](https://juliaml.github.io/MLDatasets.jl/stable/) package.
 

# Load the packages we need:

using MLDatasets: MNIST
using Flux.Data: DataLoader
using Flux: onehotbatch
using Parameters: @with_kw

# We set a default value for the size of the mini-batches:

@with_kw mutable struct Args
    minibath_size::Int = 128  ## Size of mini-batch
end

# ## Data
 
# We create the function `get_data` to get, preprare and load the data onto a DataLoader object.

function get_data(args)

    ## Load the MNIST train and test data from MLDatasets
    train_x, train_y = MNIST(:train)[:]
    test_x, test_y = MNIST(:test)[:]

    ## Reshape data to 28x28x1 multi-dimensional array
    train_x = reshape(train_x, 28, 28, 1, :)
    test_x = reshape(test_x, 28, 28, 1, :)

    ## Labels must be encoded as a vector with the same dimension 
    ## as the number of categories (unique handwritten digits) in the data set
    train_y, test_y = onehotbatch(train_y, 0:9), onehotbatch(test_y, 0:9)

    ## Now, we load the train and test images and their labels onto a DataLoader object
    data_loader_train = DataLoader(train_x, train_y, batchsize=args.minibath_size, shuffle=true)
    data_loader_test = DataLoader(train_x, train_y, batchsize=args.minibath_size, shuffle=true)

    return data_loader_train, data_loader_test

end

# This function performs the following tasks:
# * Loads the MNIST train and test images as Float32 as well as their labels. The dataset `train_x` is a 28×28×60000 multi-dimensional array. 
#   It contains 60000 elements and each one of it contains a 28x28 array. Each array represents a 28x28 image (in grayscale) of a handwritten digit. 
#   Moreover, each element of the 28x28 arrays is a pixel that represents the amount of light that it contains. On the other hand, `test_y` is a 60000 element vector and each element of this vector represents the label or actual value (0 to 9) of a handwritten digit.
# * Reshapes the train and test data to a 28x28x1 multi-dimensional array.
# * One-hot encodes the train and test labels. It creates a batch of one-hot vectors so we can pass the labels of the data as arguments for the loss function.  
# * Creates two DataLoader objects that handle data mini-batches of the size defined above.

# ## Iterate over data

# Now, we can iterate over the train data during the training routine we want to define. 

function train(; kws...)
    args = Args(; kws...)

    @info("Loading data...")
    data_loader_train, data_loader_test = get_data(args)

    ## Iterating over train data
    for (x, y) in data_loader_train
        @assert size(x) == (28, 28, 1, 128) || size(x) == (28, 28, 1, 96)
        @assert size(y) == (10, 128) || size(y) == (10, 96)
     end
end

# ## Run the example

# We call the `train` function:

cd(@__DIR__)
train()
