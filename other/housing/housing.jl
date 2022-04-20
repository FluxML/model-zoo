# # Housing data

# In this example, we create a linear regression model that predicts housing data. 
# It replicates the housing data example from the [Knet.jl readme](https://github.com/denizyuret/Knet.jl). 
# Although we could have reused more of Flux (see the MNIST example), the library's abstractions are very 
# lightweight and don't force you into any particular strategy.

# A linear model can be created as a neural network with a single layer. 
# The number of inputs is the same as the features that the data has. 
# Each input is connected to a single output with no activation function. 
# Then, the output of the model is a linear function that predicts unseen data. 

# ![singleneuron](img/singleneuron.svg)

# Source: [Dive into Deep Learning](http://d2l.ai/chapter_linear-networks/linear-regression.html#from-linear-regression-to-deep-networks)

# To run this example, we need the following packages:

using Flux
using Flux: gradient
using Flux.Optimise: update!
using DelimitedFiles, Statistics
using Parameters: @with_kw


# We set default values for the learning rate (for the training routine) and the percentage of 
# the data that we use when testing the model:

@with_kw mutable struct Hyperparams
    ## Learning rate
    lr::Float64 = 0.1 
    ## Train Test split ratio, define percentage of data to be used as Test data
    split_ratio::Float64 = 0.1 
end


# ## Data 

# We create the function `get_processed_data` to load the housing data, normalize it, 
# and finally split it into train and test datasets:


function get_processed_data(args)
    isfile("housing.data") ||
        download("https://raw.githubusercontent.com/MikeInnes/notebooks/master/housing.data",
            "housing.data")

    rawdata = readdlm("housing.data")'

    ## The last feature is our target -- the price of the house.
    split_ratio = args.split_ratio ## For the train test split

    x = rawdata[1:13,:]
    y = rawdata[14:14,:]

    ## Normalise the data
    x = (x .- mean(x, dims = 2)) ./ std(x, dims = 2)

    ## Split into train and test sets
    split_index = floor(Int,size(x,2)*split_ratio)
    x_train = x[:,1:split_index]
    y_train = y[:,1:split_index]
    x_test = x[:,split_index+1:size(x,2)]
    y_test = y[:,split_index+1:size(x,2)]

    train_data = (x_train, y_train)
    test_data = (x_test, y_test)

    return train_data,test_data
end

# This function performs the following tasks:

# 1. Downloads the housing data. The original size of the data is 505 rows and 14 columns.
# 2. Loads the data as a 14x505 matrix. This is the shape that Flux expects.
# 3. Splits the data into features and a target. Notice that the 14th row corresponds to the target for each example.
# 4. Normalizes the data. For more information on normalizing data, see [How to Use StandardScaler and MinMaxScaler Transforms in Python](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/).  
# 5. Splits the data into train and test datasets.
    

# ## Model
# We use a struct to define the model’s parameters. 
# It contains an array for holding the weights *W* and a vector for the bias term *b*:

mutable struct model
    W::AbstractArray
    b::AbstractVector
end

# Also, we create the function `predict` to compute the model’s output:

predict(x, m) = m.W*x .+ m.b

# Notice that the function `predict` takes as an argument the model struct we defined above.

# ## Loss function

# The most commonly used loss function for Linear Regression is Mean Squared Error (MSE). 
# We define the MSE function as:

meansquarederror(ŷ, y) = sum((ŷ .- y).^2)/size(y, 2)

# **Note:** An implementation of the MSE function is also available in 
# [Flux](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.mse).

# ## Train function
# Finally, we define the `train` function so that the model learns the best parameters (*W* and *b*):


function train(; kws...)
    ## Initialize the Hyperparamters
    args = Hyperparams(; kws...)
    
    ## Load the data
    (x_train,y_train),(x_test,y_test) = get_processed_data(args)
    
    ## The model
    m = model((randn(1,13)),[0.])
    
    loss(x, y) = meansquarederror(predict(x, m), y) 

    ## Training
    η = args.lr
    θ = params(m.W, m.b)

    for i = 1:500
        g = gradient(() -> loss(x_train, y_train), θ)
        for x in θ
            update!(x, g[x]*η)
        end
        if i%100==0
            @show loss(x_train, y_train)
        end
    end
    
    ## Predict the RMSE on the test set
    err = meansquarederror(predict(x_test, m),y_test)
    println(err)
end

# The function above initializes the model’s parameters *W* and *b* randomly. 
# Then, it sets the learning rate η and θ as a 
# [params object](https://fluxml.ai/Flux.jl/stable/training/training/#Flux.params) 
# that points to  W and b. Also, it sets a 
# [custom training loop](https://fluxml.ai/Flux.jl/stable/training/training/#Custom-Training-loops) 
# which is the [Gradient descent algorithm](https://en.wikipedia.org/wiki/Gradient_descent). 
# Finally, it computes the MSE for the test set.

# ## Run the example 
# We call the `train` function to run the Housing data example:

cd(@__DIR__)
train()
