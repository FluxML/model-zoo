# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
using Images
using Flux


# %%
z_dim = 2
x_dim = 512
y_dim = 512
N = 14
hidden = 9
batch_size = 1024
n = x_dim * y_dim


# %%
# cast 0:x-1 to -0.5:0.5
cast(x) = [range(-0.5, stop=0.5, step=1/(x - 1))...]


# %%
xs, ys = cast(x_dim), cast(y_dim)
xs = repeat(xs, inner=(y_dim))
ys = repeat(ys, outer=(x_dim))
rs = sqrt.(xs.^2 + ys.^2)


# %%
unit(in=N, out=N, f=tanh) = Dense(in, out, f, initW=randn)


# %%
layers = Any[unit(3 + z_dim), [unit() for _ in 1:hidden]..., unit(N, 1, σ)]

# %% [markdown]
# - In essence, CPPN is just a function, c = f(x, y), that defines the intensity of the image for every point in space.

# %%
model = Chain(layers...)
getColorAt(x) = Flux.data(model(x))

# %% [markdown]
# ## Make batches from the data

# %%
function batch(arr, s)
    l = size(arr, 2)
    batches = [arr[:, i:min(i+s-1, l)] for i=1:s:l]
    batches
end

# %% [markdown]
# ## Create image with intensities

# %%
function getImage(z)
    z = repeat(reshape(z, 1, z_dim), outer=(n, 1))
    coords = hcat(xs, ys, rs, z)'
    coords = batch(coords, batch_size)
    pixels = [Gray.(hcat(getColorAt.(coords)...))...]
    reshape(pixels, y_dim, x_dim)
end


# %%
function saveImg(z, image_path="sample.png")
    imgg = getImage(z)
    save(image_path, imgg)
    imgg
end

# %% [markdown]
# ## Will generate at random everytime

# %%
saveImg(rand(z_dim))


# %%



