using Flux

function model1()
  encoder = Chain(
    Dense(28^2, 32, relu))
  @show encoder

  decoder = Chain(
     Dense(32, 28^2, relu))
  @show decoder

  m = Chain(encoder, decoder)
  @show m
  m
end

function model2()
  encoder = Chain(
    Dense(28^2, 128, relu))
  @show encoder

  decoder = Chain(
    Dense(128, 28^2, relu))
  @show decoder

  m = Chain(encoder, decoder)
  @show m
  m
end

m = model2()
