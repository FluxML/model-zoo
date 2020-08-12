# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using CuArrays

# %% [markdown]
# # Classify MNIST digits with a simple multi-layer-perceptron
# 

# %%
imgs = MNIST.images()

# %% [markdown]
# 
# # Stack images into one large batch

# %%

X = hcat(float.(reshape.(imgs, :))...) |> gpu

labels = MNIST.labels()

# %% [markdown]
# # One-hot-encode the labels

# %%

Y = onehotbatch(labels, 0:9) |> gpu


# %%

m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax) |> gpu


# %%

loss(x, y) = crossentropy(m(x), y)


# %%

accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))


# %%

dataset = repeated((X, Y), 200)
evalcb = () -> @show(loss(X, Y))
opt = ADAM()

# %% [markdown]
# # Train

# %%

Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 10))


# %%

accuracy(X, Y)

# %% [markdown]
# 
# # Test set accuracy
# 

# %%
tX = hcat(float.(reshape.(MNIST.images(:test), :))...) |> gpu
tY = onehotbatch(MNIST.labels(:test), 0:9) |> gpu


# %%

accuracy(tX, tY)


# %%



