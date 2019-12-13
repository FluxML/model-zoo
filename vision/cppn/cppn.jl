# Ref: http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/

using Images
using Flux

# set parameters
z_dim = 2
x_dim = 512
y_dim = 512
N = 14
hidden = 9
batch_size = 1024
n = x_dim * y_dim

# cast 0:x-1 to -0.5:0.5
cast(x) = [range(-0.5, stop=0.5, step=1/(x - 1))...]

xs, ys = cast(x_dim), cast(y_dim)
xs = repeat(xs, inner=(y_dim))
ys = repeat(ys, outer=(x_dim))
rs = sqrt.(xs.^2 + ys.^2)

# sample weigths from a gaussian distribution
unit(in=N, out=N, f=tanh) = Dense(in, out, f, initW=randn)

# input -> [x, y, r, z...]
layers = Any[unit(3 + z_dim)]
for i=1:hidden
    push!(layers, unit())
end
push!(layers, unit(N, 1, σ))

model = Chain(layers...)
getColorAt(x) = model(x)

function batch(arr, s)
    batches = []
    l = size(arr, 2)
    for i=1:s:l
        push!(batches, arr[:, i:min(i+s-1, l)])
    end
    batches
end

function getImage(z)
    z = repeat(reshape(z, 1, z_dim), outer=(n, 1))
    coords = hcat(xs, ys, rs, z)'
    coords = batch(coords, batch_size)
    pixels = [Gray.(hcat(getColorAt.(coords)...))...]
    reshape(pixels, y_dim, x_dim)
end

function saveImg(z, image_path=joinpath(dirname(@__FILE__),"sample.png"))
    imgg = getImage(z)
    save(image_path, imgg)
    imgg
end

saveImg(rand(z_dim))
