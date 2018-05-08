using Flux
using Images
using Distributions

z_dim = 2
x_dim = 256
y_dim = 256
net_size = 15
net_depth = 12

# weight initialisers
init_wC = (dims...)->rand(Cauchy(0, 0.8), dims...)
init_wN = (dims...)->rand(Normal(0, 0.8), dims...)

layers = []
push!(layers, Dense(3 + z_dim, net_size, tanh))
for i=1:net_depth
    push!(layers, Dense(net_size, net_size, tanh, initW=init_wN))
end
push!(layers, Dense(net_size, 1,σ,initW=init_wN))

model = Chain(layers...)

getColorAt(x, y, r, z) = (model([x, y, r, z...]).data[1])

function getImage(z)
    # get color at each pixel
    img = []
    for i=0:(x_dim -1), j=0:(y_dim - 1)
        x, y = (i/x_dim - 0.5), (j/y_dim - 0.5)
        push!(img, getColorAt(x, y, sqrt(x^2 + y^2), z))
    end
    img
end

function showImg(z)
    imgg = Gray.(reshape(getImage(z), y_dim, x_dim))
    imgg
end

showImg(rand(Normal(0, 0.8), z_dim))
