using Images
using Flux

z_dim = 2
x_dim = 264
y_dim = 264
net_size = 15
net_depth = 5
batch_size = 512
n = x_dim * y_dim

cast(x) = (collect(0:x-1) ./ x) .- 0.5

xs, ys = cast(x_dim), cast(y_dim)
xs = repeat(xs, inner=(y_dim))
ys = repeat(ys, outer=(y_dim))
rs = sqrt.(xs.^2 + ys.^2)

layers = []
push!(layers, Dense(3 + z_dim, net_size, tanh, initW=randn))
for i=1:net_depth
    push!(layers, Dense(net_size, net_size, tanh, initW=randn))
end
push!(layers, Dense(net_size, 1, σ, initW=randn))

model = Chain(layers...)

getColorAt(x) = model(x).data

function getImage(z)
    z = repeat(reshape(z, 1, z_dim), outer=(n, 1))
    coords = hcat(xs, ys, rs, z)'
    coords = [coords[:, i:min(i + batch_size - 1, size(coords, 2))]
        for i=1:batch_size:size(coords, 2)]
    hcat(getColorAt.(coords)...)
end

function showImg(z, image_path="sample.png")
    imgg = Gray.(reshape([getImage(z)...], y_dim, x_dim))
    save(image_path, imgg)
    imgg
end

showImg(rand(z_dim))
