using Flux, Plots, Statistics
using Base.Iterators

# data generating process is autoregressive order 1
# modeling objective is to forecast y conditional
# on lags of y
function AR1(n)
    y = zeros(n)
    for t = 2:n
        y[t] = 0.9*y[t-1] + randn()
    end
    y
end    

function main()
# generate the data
n = 1000 # sample size
data = Float32.(AR1(n))
n_training = Int64(round(2*n/3))
training = data[1:n_training]
testing = data[n_training+1:end]
# set up the training
batchsize = 2   # the batchsize is the number of previous ys we look at
                # to forecast, in time series parlance, it's the lag length
epochs = 100    # maximum number of training loops through data

# the model: this is just a guess for illustrative purposes
# there may be better configurations
m = Chain(LSTM(batchsize, 10), Dense(10,2, tanh), Dense(2,batchsize))

# the first element of the batched data is one lag
# of the second element, in chunks of batchsize. The first
# elements are the inputs, the second are the outputs. So,
# we are doing one-step-ahead forecasting, conditioning
# on batchsize lags. The mod() part is to ensure that all
# batches have full size.
n = size(data,1)
training_batches = [(training[ind .- 1], training[ind]) 
    for ind in partition(2:n_training-mod(n_training,batchsize)-1, batchsize)]

# the loss function for training
function loss(x,y)
    Flux.reset!(m)
    Flux.mse(m(x)[end],y[end])
end

# function to get prediction of y conditional on lags of y.
function predict(data, batchsize)
    n = size(data,1)
    yhat = zeros(n)
    for t = batchsize+1:n
        x = data[t-batchsize:t-1]
        Flux.reset!(m)
        yhat[t] = m(x)[end]
    end
    yhat
end

# function for checking out-of-sample fit
function callback()
    error=mean(abs2.(testing[2:end]-predict(testing[1:end-1],batchsize)))
    println("testing mse: ", error)
    error
end    

# train while out-of-sample improves, saving best model.
# stop when the out-of-sample has increased too many times
bestsofar = 1e6
bestmodel = m
numincreases = 0
maxnumincreases = 5
for i = 1:epochs
    Flux.train!(loss,Flux.params(m), training_batches, ADAM())
    c = callback()
    if c < bestsofar
        bestsofar = c
        bestmodel = m
    else
        numincreases +=1
    end    
    numincreases > maxnumincreases ? break : nothing
end
m = bestmodel # use the best model found

# maximum likelihood forecast, for reference
# OLS applied to AR1 is ML estimator
y = data[2:end]
x = data[1:end-1]
ρhat = x\y
pred_ml = x*ρhat

# NN forecast
pred_nn = predict(data, batchsize)[2:end] # align with ML forecast

return pred_nn, pred_ml
end

pred_nn, pred_ml = main()
# verify that NN works as well as ML
n = size(pred_nn,1)
plot(1:n, [pred_nn pred_ml pred_nn - pred_ml], labels=["neural net forecast" "ML forecast" "difference in forecasts"])

