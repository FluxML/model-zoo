using Distributions
using Images
using Flux

z_dim = 2
x_dim = 128
y_dim = 128
net_size = 15
net_depth = 12
n = x_dim * y_dim

# weight initialiser
init_wN = (dims...)->rand(Normal(0, 0.8), dims...)

layers = []
push!(layers, Dense(3 + z_dim, net_size, tanh, initW=init_wN))
for i=1:net_depth
    push!(layers, Dense(net_size, net_size, tanh, initW=init_wN))
end
push!(layers, Dense(net_size, 1, σ, initW=init_wN))

model = Chain(layers...)

getColorAt(x) = (model(x).data)

function getImage(z)
    # get color at each pixel
    coords, img = [], []
    for i=0:(x_dim -1), j=0:(y_dim - 1)
        x, y = (i/x_dim - 0.5), (j/y_dim - 0.5)
        push!(coords, x, y, sqrt(x^2 + y^2), z...)
    end
    # process batches
    batch_size = 200
    coords = reshape(coords, 3 + z_dim, n)
    for i=1:batch_size:size(coords, 2)
        j = min(i + batch_size - 1, size(coords, 2))
        push!(img, getColorAt(coords[:, i:j])...)
    end
    img
end

function showImg(z)
    imgg = Gray.(reshape(getImage(z), y_dim, x_dim))
    save("sample.png", imgg)
    imgg
end

showImg(rand(Normal(0, 0.8), z_dim))
