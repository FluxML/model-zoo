# Inspired by "Fizz Buzz in Tensorflow" blog by Joel Grus
# http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/

using Flux
using Flux: onehotbatch, train!, setup, logitcrossentropy
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

const LABELS = ("fizz", "buzz", "fizzbuzz", "else");

# Feature engineering
features(x) = float.([x % 3, x % 5, x % 15])
features(x::AbstractArray) = reduce(hcat, features.(x))

function getdata()
    
    @test fizzbuzz.((3, 5, 15, 98)) == LABELS
    
    raw_x = 1:100;
    raw_y = fizzbuzz.(raw_x);
    
    X = features(raw_x);
    y = onehotbatch(raw_y, LABELS);
    return X, y
end

function train(; epochs::Int=500, dim::Int=20, eta::Real=0.001)

    # Get Data
    X, y = getdata()

    # Model	
    m = Chain(Dense(3 => dim, relu), Dense(dim => 4))
    loss(m, x, y) = logitcrossentropy(m(x), y)

    # Helpers
    deepbuzz(x) = (a = argmax(m(features(x))); a == 4 ? x : LABELS[a])	
	
    function monitor(e)
    	print("epoch $(lpad(e, 4)): loss = $(round(loss(m,X,y); digits=4)) | ")
        @show deepbuzz.([3, 5, 15, 98])
    end

    # Training
    opt = setup(Adam(eta), m)
    for e in 0:epochs
        if e % 50 == 0
            monitor(e) 
        end
        train!(loss, m, [(X, y)], opt)
    end

    return m
end

train()
