using Flux

info("Hooking train loop")

function Flux.train!(loss, data, opt; cb = () -> ())
  loss(first(data)...)
  opt()
  cb()
end

file(x) = joinpath(@__DIR__, "..", x)

models = [
  ("MNIST MLP","mnist/mlp.jl"),
  ("MNIST Conv","mnist/conv.jl"),
  ("MNIST Autoencoder","mnist/autoencoder.jl")]

info("Testing CPU models")
for (name, p) in models
  info(name)
  include(file(p))
end
# No GPU support
info("MNIST VAE")
info("mnist/vae.jl")

if Base.find_in_path("CuArrays") != nothing
  using CuArrays
  info("Testing GPU models")
  for (name, p) in models
    info(name)
    include(file(p))
  end
end
