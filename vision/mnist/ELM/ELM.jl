#importing important libraries
using Flux.Data.MNIST
using Flux: onehotbatch, onecold
using Statistics
using Random
"""
function elmtrain
input: 1) n * m train set; where m = #samples, n = #features
       2) c * m labels; where m = #samples, c = #classes
action: generates win and wout
"""
function elmtrain(train_x, train_y, hidden_units)
	
    #fixing layer parameters
    input_units = size(train_x,1)

    #Generating win using randn
    win = randn(hidden_units, input_units)

    #first step of forward propagation
    hidden_layer = win*train_x

    #applying ReLu
    hidden_layer = hidden_layer .* (hidden_layer .> 0)

    #calculating wout as (H^T*H)^(-1)*(H^T*Y)
    wout = (train_y*transpose(hidden_layer))*inv(hidden_layer*transpose(hidden_layer))
    global win = win
    global wout = wout
end

"""
function elmpredict
input: m * n test set; where m = #samples, n = #features
action: forward propagation and predictions
"""
function elmpredict(test_x)    
    #first step of forward propagation
    hidden_layer = win*test_x
    #applying ReLu
    hidden_layer = hidden_layer .* (hidden_layer .> 0)
    #second step of forward propagation
    output_layer = wout*hidden_layer

    #making predictions
    _, indices = findmax(output_layer, dims = 1)
    predictions = zeros(size(output_layer))

    for i in 1:size(test_x,2)
        predictions[indices[i]] = 1
    end
    return predictions
end

#loading datasets
@info "Loading Dataset"
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

#Neural network parameters
win = []
wout = []

#train using elmtrain()
@info "Training"
@time elmtrain(train_x, train_y, 1000);

#calculate training and validation accuracy
predictions = elmpredict(train_x);
Training_accuracy = mean(onecold(predictions) .== onecold(train_y));
@info "" Training_accuracy
predictions = elmpredict(test_x);
Validation_accuracy = mean(onecold(predictions) .== onecold(test_y));
@info "" Validation_accuracy
