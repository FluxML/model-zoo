# Inspired by "Fizz Buzz in Tensorflow" blog by Joel Grus
# http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/
using Flux: Chain, Dense, params, crossentropy, onehotbatch,
            ADAM, train!, softmax
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

@test fizzbuzz.([3, 5, 15, 98]) == LABELS

raw_x = 1:100;
raw_y = fizzbuzz.(raw_x);

# Feature engineering
features(x) = float.([x % 3, x % 5, x % 15])
features(x::AbstractArray) = hcat(features.(x)...)

X = features(raw_x);
y = onehotbatch(raw_y, LABELS);

# Model
m = Chain(Dense(3, 10), Dense(10, 4), softmax)
loss(x, y) = crossentropy(m(X), y)
opt = ADAM()

# Helpers
deepbuzz(x) = (a = argmax(m(features(x))); a == 4 ? x : LABELS[a])

function monitor(e)
    print("epoch $(lpad(e, 4)): loss = $(round(loss(X,y).data; digits=4)) | ")
    @show deepbuzz.([3, 5, 15, 98])
end

# Training
for e in 0:1000
    train!(loss, params(m), [(X, y)], opt)
    if e % 50 == 0; monitor(e) end
end
