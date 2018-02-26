using Flux

info("Hooking train loop")

function Flux.train!(loss, data, opt; cb = () -> ())
  loss(first(data)...)
  opt()
  cb()
end

file(x) = joinpath(@__DIR__, "..", x)

info("Testing CPU models")
info("MNIST MLP")
include(file("mnist/mlp.jl"))

if Base.find_in_path("CuArrays") != nothing
  using CuArrays
  info("Testing GPU models")
  info("MNIST MLP")
  include(file("mnist/mlp.jl"))
end
