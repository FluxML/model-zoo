# This script illustrates backpropagation through a simulation, via Flux,
# DifferentialEquations and ForwardDiff. Flux's ADAM optimiser and training loop
# are then used to optimise parameters of the simulation.

using OrdinaryDiffEq, Plots

# ––––––––––––––––––––––––––––––––––––––––––––––––––––– #
#                      ODE setup                        #
# ––––––––––––––––––––––––––––––––––––––––––––––––––––– #

# The ODE
function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)x
  du[2] = dy = (δ*x - γ)y
end

const initial_pop = 1

# Solve the ODE with a given set of parameters, to see how the predator/prey
# populations behave over time.
function trajectory(predator, prey, tfinal = 10)
  params = [predator..., prey...]
  T = eltype(params)
  u0 = T.([initial_pop, initial_pop])
  tspan = (T(0), T(tfinal))
  prob = ODEProblem(lotka_volterra, u0, tspan, params)
  solve(prob, Tsit5(), dtmin = 1e-4)
end

# See an example solution
plot(trajectory([1.8, 1.5], [1.2, 3]), ylim=(0,6))

# For now, our loss is the deviation from the initial population;
# we are optimising for stability.
function stability(sol::ODESolution)
  sol.retcode != :Success && return zero(sol.u[1][1])
  series = sol.(range(0, stop = 10, length = 50))
  sum(x -> sum(x -> (x - initial_pop)^2, x), series)/length(series)
end

stability(predator, prey, tfinal = 100) =
  stability(trajectory(predator, prey, tfinal))

# Preview the loss
stability([1.8, 1.5], [1.2, 3])

# ––––––––––––––––––––––––––––––––––––––––––––––––––––– #
#                      Autodiff                         #
# ––––––––––––––––––––––––––––––––––––––––––––––––––––– #

using Flux
import Flux.Tracker: TrackedVector, @grad, track
using ForwardDiff: Dual, value, partials

# Provide a custom gradient for `stability`, which will use forward-mode AD
# internally.
stability(predator::TrackedVector, prey) = track(stability, predator, prey)

@grad function stability(predator, prey)
  s = stability(Tracker.data(predator) .+ [Dual(0,1,0),Dual(0,0,1)], prey)
  value(s), Δ -> (Δ .* partials(s), nothing)
end

# Now we can take gradients w.r.t. the parameters.

pred = [2.2, 1.0]
prey = [2, 3]
Tracker.derivative(p -> stability(p, prey), pred)

# ––––––––––––––––––––––––––––––––––––––––––––––––––––– #
#               Optimising Parameters                   #
# ––––––––––––––––––––––––––––––––––––––––––––––––––––– #

predator = param([2.2, 1.0])
prey = [2, 3]

data = Iterators.repeated((), 100)
opt = ADAM([predator], 0.1)
cb = () ->
  display(plot(trajectory(Flux.data(predator), prey), ylim=(0,6)))

# Display the ODE with the current parameter values.
cb()

# Running this in Juno will generate an animation of the ODE over time.
Flux.train!(() -> stability(predator, prey),
            data, opt, cb = cb)
