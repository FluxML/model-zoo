# # Iris data

# In this example, we create a logistic regression model that classifies iris flowers. 
# It consists of a [single-layer neural network](https://sebastianraschka.com/faq/docs/logisticregr-neuralnet.html) 
# that outputs **three** probabilities (one for each species of iris flowers). 
# We use Fisher's classic dataset to train the model. This dataset is retrieved from 
# the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

# In Machine Learning, a classification task can be performed by a logistic regression model. 
# However, we also can create a logistic regression model as a single-layer neural network. 
# This neural network has the following characteristics:

# * Uses the [logitcrossentropy](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.logitcrossentropy) loss function.
# * Expects the class labels of the iris flowers encoded using [One-Hot encoding](https://fluxml.ai/Flux.jl/stable/data/onehot/#One-Hot-Encoding).
# * Outputs the index in the output vector with the highest value as the class label using 
# [onecold](https://fluxml.ai/Flux.jl/stable/data/onehot/#Flux.onecold) which is the inverse operation of One-Hot encoding.

# To run this example, we need the following packages:

# Suggested in the documentation readme, but uncomment if installation of packages is needed
# import Pkg 
# Pkg.activate(".") # activate in the folder of iris
# Pkg.instantiate() # installs required packages for the example

using Flux, MLDatasets, DataFrames
using Flux: logitcrossentropy, normalise, onecold, onehotbatch
using Statistics: mean

# We set default values for the learning rate *lr* (for the training routine) and the number of
# times that we repeat the train data (more information below):

Base.@kwdef mutable struct Args
    lr::Float64 = 0.5
    repeat::Int = 110
end

# ## Data

# We create the function `get_processed_data` to load the iris data, preprocess 
# it (normalize and One-Hot encode the class labels), and split it into train and test datasets.


function get_processed_data(args::Args)

    iris = Iris(as_df=false)
    labels = iris.targets |> vec
    features = iris.features 

    ## Subract mean, divide by std dev for normed mean of 0 and std dev of 1.
    normed_features = normalise(features, dims=2)

    klasses = sort(unique(labels))
    onehot_labels = onehotbatch(labels, klasses)

    ## Split into training and test sets, 2/3 for training, 1/3 for test.
    train_indices = [1:3:150 ; 2:3:150]

    X_train = normed_features[:, train_indices]
    y_train = onehot_labels[:, train_indices]

    X_test = normed_features[:, 3:3:150]
    y_test = onehot_labels[:, 3:3:150]

    ## Repeat the data `args.repeat` times
    train_data = Iterators.repeated((X_train, y_train), args.repeat)
    test_data = (X_test,y_test)

    return train_data, test_data
end

# The iris data is a 4×150 matrix. This means that the iris data has 150 examples, 
# and each example has four features as well as a class label. 
# After normalizing and encoding the data, the `get_processed_data` function divides it into train and test data. 
# Also, it repeats the examples in the train data so that we have more data to train the neural network.


# ## Metrics

# We use two functions to assess the output of the model: `accuracy` and `confusion matrix`. 
# The [accuracy function](https://developers.google.com/machine-learning/crash-course/classification/accuracy) 
# measures the percentage of the labels that the model classified correctly. 
# On the other hand, the [confusion matrix](https://machinelearningmastery.com/confusion-matrix-machine-learning/) 
# is a table that summarises how good the model is for predicting data. 


accuracy(model, x, y) = mean(onecold(model(x)) .== onecold(y))


function confusion_matrix(model, X, y)
    ŷ = onehotbatch(onecold(model(X)), 1:3)
    y * transpose(ŷ)
end

# ## Train function

# We define the `train` function that defines the model and trains it:

function train(; kws...)
    ## Initialize hyperparameter arguments
    args = Args(; kws...)	

    ## Load processed data
    train_data, test_data = get_processed_data(args)

    ## #Declare model taking 4 features as inputs and outputting 3 probabiltiies, 
    ## one for each species of iris.
    model = Chain(Dense(4, 3))
	
    ## Define loss function to be used in training
    ## For numerical stability, we use here logitcrossentropy
    loss(m, x, y) = logitcrossentropy(m(x), y)
	
    ## Training
    ## Gradient descent optimiser with learning rate `args.lr`
    optimiser = Descent(args.lr)
    ## For any other optimiser, we would need e.g. 
    ## opt_state = Flux.setup(Momentum(args.lr), model)

    println("Starting training.")
    Flux.train!(loss, model, train_data, optimiser)
	
    return model, test_data
end

# The function above loads the train and test data. 
# Then, it creates the model as a single-layer network that expects as an input 
# a four-element vector (features) and outputs a three-element vector 
# (the number of classes of species of iris flowers). 
# Also, it sets [logitcrossentropy](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.logitcrossentropy) 
# as the loss function and the Gradient descent optimiser 
# [Descent](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.Descent). 
# Finally, it runs a training step with the 
# [train! function](https://fluxml.ai/Flux.jl/stable/training/training/#Flux.Optimise.train!).

# ## Test function

# After training the model, we define the `test` function that 
# computes the model performance on the test data. 
# It calls the `accuracy` function and displays the confusion matrix. 
# This function validates that the model should achieve at least a 0.8 accuracy score.


function test(model, test)
    ## Testing model performance on test data 
    X_test, y_test = test
    accuracy_score = accuracy(model, X_test, y_test)

    println("\nAccuracy: $accuracy_score")

    ## Sanity check.
    @assert accuracy_score > 0.8

    ## To avoid confusion, here is the definition of a 
    ## Confusion Matrix: https://en.wikipedia.org/wiki/Confusion_matrix
    println("\nConfusion Matrix:\n")
    display(confusion_matrix(model, X_test, y_test))
end

# ## Run the example

# We call the `train` function to run the iris data example and compute the model performance:

cd(@__DIR__)
model, test_data = train()
test(model, test_data)
