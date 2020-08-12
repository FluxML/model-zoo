# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
using Flux, Flux.Data.MNIST
using Flux: @epochs, onehotbatch, mse, throttle
using Base.Iterators
using CuArrays
using Images

# %% [markdown]
# ## Encode MNIST images as compressed vectors that can later be decoded back into images.
# 

# %%

imgs = MNIST.images()

# %% [markdown]
# # Partition into batches of size 1000

# %%
data = [float(hcat(vec.(imgs)...)) for imgs in partition(imgs, 1000)]
data = gpu.(data)


# %%
N = 32 # Size of the encoding

# %% [markdown]
# - You can try to make the encoder/decoder network larger
# - Also, the output of encoder is a coding of the given input.
# - In this case, the input dimension is 28^2 and the output dimension of
# - encoder is 32. This implies that the coding is a compressed representation.
# - We can make lossy compression via this `encoder`.

# %%
encoder = Dense(28^2, N, leakyrelu) |> gpu
decoder = Dense(N, 28^2, leakyrelu) |> gpu


# %%
m = Chain(encoder, decoder)


# %%
loss(x) = mse(m(x), x)


# %%

evalcb = throttle(() -> @show(loss(data[1])), 5)
opt = ADAM()

# %% [markdown]
# ## Train

# %%
@epochs 10 Flux.train!(loss, params(m), zip(data), opt, cb = evalcb)

# %% [markdown]
# # Sample output

# %%
img(x::Vector) = Gray.(reshape(clamp.(x, 0, 1), 28, 28))


# %%
function sample()
  # 20 random digits
  before = [imgs[i] for i in rand(1:length(imgs), 20)]
  # Before and after images
  after = img.(map(x -> cpu(m)(float(vec(x))).data, before))
  # Stack them all together
  hcat(vcat.(before, after)...)
end


# %%
save("sample.png", sample())


