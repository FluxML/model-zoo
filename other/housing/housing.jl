# # Machine Learning Problem : Housing Dataset
#
# The housing problem functions as a starting point in Machine Learning.
# We'll be demonstrating the use of Julia's [Flux Package](https://fluxml.ai/)
# to do this problem.
#
# The data replicates the housing data example from the Knet.jl readme. Although we
# could have reused more of Flux (see the mnist example), the library's
# abstractions are very lightweight and don't force you into any particular
# strategy.
# 
# [This](http://www.mit.edu/~6.s085/notes/lecture3.pdf) might help you know more about the fundamentals of what 
# we're about to do. If you don't understand something there which is also not mentioned here in this file, 
# you may overlook that (or search it up on google to quench your curiosity :-)

using Flux.Tracker, Statistics, DelimitedFiles
using Flux.Tracker: Params, gradient, update!
using DelimitedFiles, Statistics
using Flux: gpu

# ## Getting the data and other pre-processing.
# We'll start by getting <code>housing.data</code> and splitting it into 
# training and test sets. 
# Training Dataset is the sample of data used to **fit** the model while
# Test Dataset is the sample of data used to provide an unbiased evaluation 
# of a final model fit on the training dataset.

# Our aim is to predict the price of the house. In this dataset, the last
# feature is the price and would therefore be our target.

cd(@__DIR__)

isfile("housing.data") ||
  download("https://raw.githubusercontent.com/MikeInnes/notebooks/master/housing.data",
           "housing.data")

rawdata = readdlm("housing.data")'

#-

# Specifying the split ratio and **x** and **y**
split_ratio = 0.1 

x = rawdata[1:13,:] |> gpu
y = rawdata[14:14,:] |> gpu

# ### Normalising
# What is the need ? 
# Normalization is a technique often applied as part of data preparation for machine learning. 
# The goal of normalization is to change the values of numeric columns in the dataset to a common scale,
# without distorting differences in the ranges of values. For machine learning, every dataset does not require normalization.
# It is required only when features have different ranges like in this case.

x = (x .- mean(x, dims = 2)) ./ std(x, dims = 2)

# ### Splitting into test and training sets.

split_index = floor(Int,size(x,2)*split_ratio)
x_train = x[:,1:split_index]
y_train = y[:,1:split_index]
x_test = x[:,split_index+1:size(x,2)]
y_test = y[:,split_index+1:size(x,2)]

# ## The Model 
# Here comes everyone's favourite part : implementing a machine learning model.
# 
# A ML model is in it's simplest terms a mathematical model which has a number of parameters
# that need to be learned from the data provided. The data has an important task: to fit our model parameters.
# The more data we have, the more we can accurately predict the target.
#
# Hyperparameters aren't learnt during the training process. They can be treated as constants that are fixed for the 
# entire process. These parameters express important properties of the model such as its complexity or how fast it should learn.
#
# We'll now define the Weight (W) and the Bias (b) terms. They are our hyperparameter which
# we tune to enhance our predictions during gradient descent. 
# To get an intution about how gradientDescent actually works, check out Andrew Ng's awesome explaination 
# here. [Video 1: Intution](https://www.youtube.com/watch?v=rIVLE3condE) | 
# [Video 2: The Algorithm](https://www.youtube.com/watch?v=yFPLyDwVifc)

W = param(randn(1,13)/10) |> gpu
b = param([0.]) |> gpu

# Here are our prediction and loss functions.
# - The prediction functions returns our prediction of the price of the house as 
# suggested by our 2 hyperparameters: W and b.
# - MSE is the average of the squared error that is used as the loss function for least squares regression.
# It is defined as the sum, over all the data points, of the square of the difference between the predicted and actual target
# variables, divided by the number of data points. 
#
# Loss functions evaluate how well your algorithm models your dataset. 
# If predictions are off, the loss function is high. If they're good, it'll be low.

predict(x) = W*x .+ b
meansquarederror(ŷ, y) = sum((ŷ .- y).^2)/size(y, 2)
loss(x, y) = meansquarederror(predict(x), y)

# ### Gradient Descent 
# Optimizing our parameters to get accurate prediction. Learn more from the links I mentioned above.

η = 0.1
θ = Params([W, b])

for i = 1:10
  g = gradient(() -> loss(x_train, y_train), θ)
  for x in θ
    update!(x, -g[x]*η)
  end
  @show loss(x_train, y_train)
end

# ## Predictions
# Now we're in a position to know how well our program works on the given data.

err = meansquarederror(predict(x_test),y_test)
println(err)

# The prepared model might not very good for predicting the housing prices and may have high error.
# One can improve the prediction results using many other possible machine learning algorithms and techniques.
# If this was your first ML project in Flux, Congrats! 
# 
# You should have gotten a gist of basic ML functionality in Flux Package using Julia by now.

# ## References : 
# 1. [Introduction to Loss Functions](https://algorithmia.com/blog/introduction-to-loss-functions)
# 2. [Why Data Normalization is necessary for Machine Learning models](https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029)
# 3. [About Train, Validation and Test Sets in Machine Learning](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7)
# 4. [How to select the Right Evaluation Metric for Machine Learning Models: Part 1 Regression Metrics](https://towardsdatascience.com/how-to-select-the-right-evaluation-metric-for-machine-learning-models-part-1-regrression-metrics-3606e25beae0)
# 5. [MIT's Notes on Linear Regression](http://www.mit.edu/~6.s085/notes/lecture3.pdf)
# 6. [ML | Hyperparameters: An Understanding](https://www.geeksforgeeks.org/ml-hyperparameter-tuning/)
#
# And lastly, the course to which I owe this understanding:
# [Stanford's Machine Learning](https://www.coursera.org/learn/machine-learning) 
# as taught by Andrew Ng.

