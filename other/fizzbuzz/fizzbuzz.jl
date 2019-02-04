# Inspired by "Fizz Buzz in Tensorflow" blog by Joel Grus
# http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/
using Flux: Chain, Dense, params, crossentropy, onehotbatch,
            ADAM, train!, softmax

using Random
using Test

# Data preparation
function fizzbuzz(x::Int)
    is_divisible_by_three = x % 3 == 0
    is_divisible_by_five = x % 5 == 0

    if is_divisible_by_three & is_divisible_by_five
        return "fizzbuzz"
    elseif is_divisible_by_three
        return "fizz"
    elseif is_divisible_by_five
        return "buzz"
    else
        return "else"
    end
end

const LABELS = ["fizz", "buzz", "fizzbuzz", "else"];



raw_x = 100:10000; #Made training dataset larger and exclusive of test set
raw_y = fizzbuzz.(raw_x);

# Feature engineering
features(x) = float.([x%3 , x%5 , x%15 ])
features(x::AbstractArray) = hcat(features.(x)...)

X = features(raw_x);
y = onehotbatch(raw_y, LABELS);

# Model
m = Chain(Dense(3, 100), Dense(100, 4), softmax)
loss(x, y) = crossentropy(m(X), y)
opt = ADAM()

# Helpers
deepbuzz(x) = (a = argmax(m(features(x))); a = LABELS[a])

function monitor(e)
    print("epoch $(lpad(e, 4)): loss = $(round(loss(X,y).data; digits=4)) | ")
    @show deepbuzz.([3, 5, 15, 98])
end

# Training
for e in 0:1000
    #Randomizing training data to avoid sequential bias
    Random.shuffle!(float.(raw_x))
    raw_y = fizzbuzz.(raw_x)
    X = features(raw_x);
    y = onehotbatch(raw_y, LABELS);
    train!(loss, params(m), [(X, y)], opt)
    if e % 50 == 0; monitor(e) end
end
#Test Cases added
@test fizzbuzz.([3, 5, 15, 98]) == deepbuzz.([3,5,15,98])
@test fizzbuzz.([10,9,60, 99]) == deepbuzz.([10,9,60,99])
@test fizzbuzz.([35, 42, 99, 3]) == deepbuzz.([35,42,99,3])
@test fizzbuzz.([55,64,72,18]) == deepbuzz.([55,64,72,18])