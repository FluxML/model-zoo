using Flux, Flux.Tracker
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated
using CuArrays, CUDAnative
################################################################################
# Decoupled Neural Interface consists of two parts: A feedforward neural network
# and a neural network to compute gradients. `net` is the FFNN here and dni is
# NN to compute gradients

struct SynthMLP
  net::Chain
  dni::Chain
  conditioned::Bool
  opt
end


function SynthMLP(input_dim::Int, output_dim::Int, hidden_dims = [];
  act_fns = [identity], conditioned = true)

  forward = []
  forward_dni = []
  dims = push!(append!([input_dim], hidden_dims), output_dim)

  for i = 1:length(dims)-1
    push!(forward, Dense(dims[i], dims[i+1], act_fns[i]))
    d = dims[i+1] + output_dim * conditioned
    push!(forward_dni, Dense(d, dims[i+1]))
  end

  net = Chain(forward...) |> gpu
  dni = Chain(forward_dni...) |> gpu

  SynthMLP(net, dni, conditioned, ADAM(vcat(params(net),  params(dni))))
end

(s_mlp::SynthMLP)(x) = s_mlp.net(x)
#################### Training a layer using Synthetic Gradients ################
function train_layer(layer, dni_layer, x, y = nothing; conditioned=false)
  inp = param(x)
  out = layer(inp)

  # Concatenate labels to DNI input if DNI is conditioned on labels
  if conditioned
    @assert size(y, 2) == size(out, 2)
    dni_inp = vcat(out, param(y))
  else
    dni_inp = out
  end
  # This is predicted gradient of `out`; ie dLoss / dout
  grad = dni_layer(dni_inp)
  # Backpropagating the prediected grads to the Weights and input
  Flux.back!(out, grad.data)
  #returning the output, ∇̂output, ∇input
  out.data, grad, inp.grad
end

function train_net(s_mlp::SynthMLP, x, y = nothing; loss = (a,b)->crossentropy(softmax(a),b),
  synthLoss = (a,b)->Flux.mse(a,b))
  out = x
  dĥ, _dh = [], [] # predicted grads of output of a layer
                   # true grads of input to a layer

  for (layer, dni_layer) in zip(s_mlp.net.layers, s_mlp.dni.layers)
    out, Δ, _Δ = train_layer(layer, dni_layer, out, y; conditioned = s_mlp.conditioned)
    push!(dĥ, Δ)
    push!(_dh, _Δ)
  end

  # Taking true gradient of final output
  cc, back = Tracker.forward((a, b)->loss(a, b), out, y)

  dout, _ = Tracker.data.(back(ones(out))) |> gpu

  # Gradient of the input data to the network is not required
  dh = push!(_dh[2:end], dout)
  for (dŷ, dy) in zip(dĥ, dh)
    # Taking Synthetic loss for training the DNI
    @show l = synthLoss(dŷ, dy)
    Flux.back!(l)
  end

  # optimising the parameters using accumulated grads
  s_mlp.opt()
  
  loss(out, y)
end

################################################################################

using Flux.Data.MNIST

imgs = MNIST.images()
# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...) |> gpu

labels = MNIST.labels()
# One-hot-encode the labels
Y = Float32.(onehotbatch(labels, 0:9)) |> gpu

dataset = repeated((X, Y), 200)

model = SynthMLP(784, 10, [256], act_fns = [relu, identity])

accuracy(x, y) = mean(argmax(model(x)) .== argmax(y))

epochs = 1
for (x, y) in dataset
  l = train_net(model, x, y)
  acc = accuracy(x, y)
  println("Epoch $epochs over. Loss: $l. Accuracy: $acc")
  epochs += 1
end
