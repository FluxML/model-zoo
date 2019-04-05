include("automata.jl")

using Flux, Zygote, Statistics
using Zygote: @adjoint

activate(x) = x > 0.5
@adjoint activate(x) = activate(x), ȳ -> (ȳ,)

radius = 3

model = Chain(Dense(2radius+1, 10, σ), Dense(10, 1, σ))

automata(st, i) = model(neighbourhood(st, i, radius))[1] |> activate

function loss(st, steps = length(st))
  target = vote(st)
  for i = 1:steps
    st = step(st, automata)
  end
  (target - mean(st.st))^2
end

st = State(500)

loss(st)
image(st, automata)

gradient(loss, st)

gs = gradient(() -> loss(st), params(model))
