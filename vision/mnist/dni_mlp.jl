using Flux, Flux.Data.MNIST
using Flux: onehotbatch, argmax, crossentropy
using CuArrays

################################################################################
# Decoupled Neural Interface consists of two parts: A feedforward neural network
# and a neural network to compute gradients. `net` is the FFNN here and dni is
# NN to compute gradients

struct SynthMLP
  net::Chain
  dni::Chain
  conditioned::Bool
  opt_net
  opt_dni
end


function SynthMLP(input_dim::Int, output_dim::Int, hidden_dims = [];
  act_fns = [identity], conditioned = true)

  forward = []
  forward_dni = []
  opt_net, opt_dni = [], []
  dims = push!(append!([input_dim], hidden_dims), output_dim)
  
  for i = 1:length(dims)-1
    push!(forward, Dense(dims[i], dims[i+1], act_fns[i]))   
    d = dims[i+1] + output_dim * conditioned
    push!(forward_dni, Dense(d, dims[i+1], initW = zeros))
  end

  net = Chain(forward...) |> gpu
  # Last layer does not require synthetic grads, it can directly obtain true grads
  dni = Chain(forward_dni[1:end-1]...) |> gpu

  for (layer, dni_layer) in zip(net.layers[1:end-1], dni.layers)
    push!(opt_net, ADAM(params(layer)))
    push!(opt_dni, ADAM(params(dni_layer)))
  end
  
  push!(opt_net, ADAM(params(net.layers[end])))

  SynthMLP(net, dni, conditioned, opt_net, opt_dni)
end

(s_mlp::SynthMLP)(x) = s_mlp.net(x)

#################### Training a layer using Synthetic Gradients ################

function train_layer(layer, dni_layer, opt_layer, x, y = nothing; conditioned=false)
  inp = param(x)
  out = layer(inp)

  # Concatenate labels to DNI input if DNI is conditioned on labels
  if conditioned
    @assert size(y, 2) == size(out, 2)
    dni_inp = vcat(out.data, y)
  else
    dni_inp = out.data
  end
  # This is predicted gradient of `out`; ie dLoss / dout
  grad = dni_layer(dni_inp)
  # Backpropagating the predicted grads to the Weights and input
  Flux.back!(out, grad.data)  
  
  opt_layer()
  
  #returning the output, ∇̂output, ∇input
  out.data, grad, inp.grad
end

# Last layer does not use any synthetic gradients
function train_last_layer(layer, opt_layer, x, y, loss)
  inp = param(x)
  out = layer(inp)

  l = loss(out, y)
  Flux.back!(l)

  opt_layer()

  out.data, inp.grad, l.data
end

function train_net(s_mlp::SynthMLP, x, y = nothing; loss = (a,b)->crossentropy(softmax(a), b),
  synthLoss = (a,b)->Flux.mse(a,b))
  out = x
  dĥ, _dh = [], [] # predicted grads of output of a layer
                   # true grads of input to a layer

  for (layer, dni_layer, opt_layer) in zip(s_mlp.net.layers[1:end-1], s_mlp.dni.layers, s_mlp.opt_net[1:end-1])
    out, Δ, _Δ = train_layer(layer, dni_layer, opt_layer, out, y; conditioned = s_mlp.conditioned)				    
    push!(dĥ, Δ)
    push!(_dh, _Δ)
  end
  
  out, _Δ, loss_ = train_last_layer(s_mlp.net.layers[end], s_mlp.opt_net[end], out, y, loss)

  # Gradient of the input data to the network is not required
  dh = push!(_dh[2:end], _Δ)
  
  for (dŷ, dy, opt_dni_layer) in zip(reverse.([dĥ, dh, s_mlp.opt_dni])...)
    # Taking Synthetic loss for training the DNI
    l = synthLoss(dŷ, dy)
    Flux.back!(l)  
    opt_dni_layer()
  end
  
  loss_
end

################################################################################

imgs = MNIST.images()

# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...) |> gpu
Y = float.(onehotbatch(MNIST.labels(), 0:9)) |> gpu

model = SynthMLP(784, 10, [256]; act_fns = [relu, identity])

accuracy(x, y) = mean(argmax(model(x)) .== argmax(y))

for epoch = 1:200 
  l   = train_net(model, X, Y)
  acc = accuracy(X, Y)
  println("Epoch: $epoch. Loss: $l. Accuracy: $acc")
end


# Let's check the accuracy on testset

tX = hcat(float.(reshape.(MNIST.images(:test), :))...) |> gpu
tY = onehotbatch(MNIST.labels(:test), 0:9) |> gpu

acc = accuracy(tX, tY)
println("Accuracy on testset: $acc")
