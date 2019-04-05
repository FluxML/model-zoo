include("automata.jl")

using Flux, Zygote
using Zygote: @adjoint

activate(x) = x > 0.5
@adjoint activate(x) = activate(x), ȳ -> (ȳ,)

radius = 3

model = Chain(Dense(2radius+1, 10, relu), Dense(10, 1, σ))

automata(st, i) = model(neighbourhood(st, i, radius))[1] |> activate

automata(State(5), 1)

st = State(500)
image(st, automata)

automata(st, 1)

gradient(automata, st, 1)

gs = gradient(() -> automata(st, 1), params(model))
