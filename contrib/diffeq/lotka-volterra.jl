# WARNING:
# ========
# This blog post https://julialang.org/blog/2019/01/fluxdiffeq/ gives a high level
# description of this file.

# Many function calls are fully qualified. This is not necessary. The package name is
# included to get a better sense of the role of the different packages.

# The Lotka-Volterra differential equations model populations of predators
# and preys over time.

# This example starts with an ODE setup with a given set of parameters.
# Then, using the same equations, optimises starting with a different set of parameters

## Import the required packages
using DifferentialEquations, DiffEqSensitivity
using Flux, DiffEqFlux
using Plots


##
## Setup ODE to optimize
##

# The populations are representated as a vector of 2 numbers:
#    U = [numbers of rabbits, number of wolves]
#
# ODE problems expect the equations to be presented with standard parameters:
#   dU: the derivatives of U. This vector is modified in place for speed purposes
#    U: the functions
#    p: a vector of paramters
#    t: the functions' variable (here time)
function lotka_volterra!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α * x - β * x * y
  du[2] = dy = -δ * y + γ * x * y
end

# Initial conditions
u0 = [1.0, 1.0]

# Time over which to simulate
t_span = (0.0, 10.0)

# Initial parameters
p0 = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE
prob0 = DiffEqBase.ODEProblem(lotka_volterra!, u0, t_span, p0)

# Solve the ODE
sol0 = DiffEqBase.solve(prob0, Tsit5())

# Plot it
plot(sol0)

##
## Generate data that will be used to train the network
## The data is the number of rabbits/wolves at set times
##

# Solve the ODE (again) and collect solutions at fixed intervals
target_data = DiffEqBase.solve(prob0, Tsit5(), saveat = 0.1)
rabbits = target_data[1, :]                     # vector of 101 data points
wolves  = target_data[2, :]                    # vector of 101 data points

# Plot the data on top of the full solution
t_steps = 0:0.1:10.0
scatter!(t_steps, rabbits, color = [1], label = "rabbits")
scatter!(t_steps, wolves,  color = [2], label = "wolves")


##
## Parameter optimisation
##

# Loss function is the total squared error
# (the number of points is constant - mean is not necessary)
loss_function = function()
    prediction = DiffEqBase.concrete_solve(prob0, Tsit5(), u0, p;
    sensealg = TrackerAdjoint(), saveat = 0.0:0.1:10.0)

    # Calculate squared error
  return sum(abs2, prediction - target_data)
end


# Callback function to observe training
iter = 0
callback = function ()
  global iter += 1
  if iter % 10 == 1
    # Plot the training data
    scatter(t_steps,  rabbits, color = [1], label = "rabbit data")
    scatter!(t_steps, wolves,  color = [2], label = "wolves data")

    # Use `remake` to re-create the original `prob0` with different current parameters `p`
    remade_sol = DiffEqBase.solve(remake(prob0, p = p), Tsit5(), saveat = 0.1)
    remade_rabbits = remade_sol[1, :]
    remade_wolves  = remade_sol[2, :]
    plot!(t_steps, remade_rabbits, ylim = (0, 6), labels = "rabbit model", color = 1)
    display(plot!(t_steps, remade_wolves, ylim = (0, 6), labels = "wolf model", color = 2))
  end
end


# Vector of new parameters different from p0
p = [4.0, 1.0, 2.0, 0.4]
ODEparams = Flux.params(p)

# Display the ODE with the initial parameter values.
callback()


# The parameter `loss_data` in `train!` (below) contains a list of data that are provided
# to the loss function. `train!`` works by iterating over that list. In our case, the
# calculation of the loss function does not require any additional data.
# Therefore `loss_data` is filled with a whole bunch of nothing.
loss_data = Iterators.repeated((), 1000)

# ADAM is a great default optimiser
optimiser = Flux.ADAM(0.1)

# train! is a function that hides some of the complexity of the library
Flux.train!(loss_function, ODEparams, loss_data, optimiser; cb = callback)
