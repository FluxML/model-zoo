#importing important libraries
using Flux, Flux.Data.MNIST
using Flux: onehotbatch, onecold
using Statistics
using Random

#loading datasets
data = MNIST.images();

#flattening training set
X = hcat(float.(reshape.(data, :))...);

#loading labels, shuffling the dataset and applying onehot
labels = MNIST.labels()
X = [X; transpose(labels)];
X = X[:, shuffle(1:end)];
labels = X[end, :];
X = X[1:end-1, :];
Y = onehotbatch(labels, 0:9);

#splitting datasets into train and test and test set
split_index = convert(Int32, 0.8*size(X,2))
train_x = X[:, 1:split_index];
train_y = Y[:, 1:split_index];
test_x = X[:, split_index+1:size(X,2)];
test_y = Y[:, split_index+1:size(X,2)];

#loading Elm module
include("ELM.jl")
using .Elm

#train using elmtrain()
elmtrain(train_x, train_y, 1000);

#calculate training and validation accuracy
predictions = elmpredict(train_x);
accuracy = mean(onecold(predictions) .== onecold(train_y));
println("Training accuracy   = ", accuracy)
predictions = elmpredict(test_x);
accuracy = mean(onecold(predictions) .== onecold(test_y));
println("Validation accuracy = ", accuracy)
