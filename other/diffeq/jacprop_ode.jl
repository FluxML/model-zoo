# In this script, we explore the effects of tangent-space regularization by means of Jacobian propagation (jacprop) for smooth dynamical systems, as proposed in
# "Machine Learning and System Identification for Estimation in Physical Systems", Bagge Carlson, F. chap 8.
# This file is very similar to the example [neural_ode.jl](neural_ode.jl), but adds a model that is trained using tangent-space regularization to promote a smooth dynamics function.

using Flux, DiffEqFlux, DifferentialEquations, Plots, IterTools
using Flux: data, throttle, train!
using Flux.Tracker: jacobian

Random.seed!(123)
batchsize = 50
u0        = [randn(Float32, 2) for _ in 1:batchsize]
datasize  = 30
tspan     = (0.0f0,1.5f0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
function trueODEfunc(u) # We define this function as well to easily calculate the true jacobian
    true_A = [-0.1 2.0; -2.0 -0.1]
    ((u.^3)'true_A)'
end

t = range(tspan[1],tspan[2],length=datasize)
ode_data = map(u0) do u0
  prob = ODEProblem(trueODEfunc,u0,tspan)
  Array(solve(prob,Tsit5(),saveat=t))
end

# Define two identical models, one for normal training and one for trainging with jacprop.
dudt = Chain(x -> x.^3,
             Dense(2,50,tanh),
             Dense(50,2))
dudtj = Chain(x -> x.^3,
             Dense(2,50,tanh),
             Dense(50,2))


truejacobian(x) = Flux.data(Flux.Tracker.jacobian(trueODEfunc, x))
ps     = Flux.params(dudt)
psj    = Flux.params(dudtj)
n_ode  = x->neural_ode(dudt,x,tspan,Tsit5(),saveat=t,reltol=1e-5,abstol=1e-7)
n_odej = x->neural_ode(dudtj,x,tspan,Tsit5(),saveat=t,reltol=1e-5,abstol=1e-7)

pred = n_ode(u0[1]) # Get the prediction using the correct initial condition
scatter(t,ode_data[1][1,:],label="data")
scatter!(t,Flux.data(pred[1,:]),label="prediction")
##

loss_n_ode(x,y)  = mean(abs2,y .- n_ode(x))
loss_n_odej(x,y) = mean(abs2,y .- n_odej(x))
λ = 10 # Tuning parameter to determine the amount of tangent-space regularization. A higher value makes the dynamics function smoother along a trajectory
function loss_jacprop(x,y)
  T  = size(y,2)
  l  = loss_n_odej(x,y)
  J1 = jacobian(x->dudtj[2:end](x), dudt[1](y[:,1])) # Since the input nonlinearity (x^3) was assumed known, we penalize the change in the learnable jacobian only.
  for i in 2:T
    J2 = jacobian(x->dudtj[2:end](x), dudt[1](y[:,i]))
    l += λ*sum(abs2, J1-J2)/T
    J1 = J2
  end
  l
end

dataset = IterTools.ncycle(zip(u0, ode_data), 30)
opt     = RMSProp(0.01)
optj    = RMSProp(0.01)
cb = function () #callback function to observe training
  display(sum(x->loss_n_ode(x...), Iterators.take(dataset, 3)))
  # plot current prediction against data
  cur_pred = data(n_ode(u0[1]))
  pl = scatter(t,ode_data[1][1,:],label="data")
  scatter!(pl,t,cur_pred[1,:],label="prediction")
  display(plot(pl))
end
cbj = function () #callback function to observe training
  display(sum(x->loss_n_odej(x...), Iterators.take(dataset, 3)))
  # plot current prediction against data
  cur_pred = data(n_odej(u0[1]))
  pl = scatter(t,ode_data[1][1,:],label="data")
  scatter!(pl,t,cur_pred[1,:],label="prediction")
  display(plot(pl))
end


println("Training normally")
train!(loss_n_ode, ps, dataset, opt, cb = throttle(cb,15))
println("Training with jacprop")
train!(loss_jacprop, psj, dataset, optj, cb = throttle(cbj,15))

prob = ODEProblem(trueODEfunc,Float32[2,0],tspan)
yval = Array(solve(prob,Tsit5(),saveat=t)) # Form a validation trajectory

jacmat(x)   = reduce(hcat, vec.(data.(x)))'
truejacs    = map(truejacobian, eachcol(yval))
normaljacs  = map(x->jacobian(dudt,x), eachcol(yval))
jacpropjacs = map(x->jacobian(dudtj, x), eachcol(yval))
primary     = [true false false false]
plot(jacmat(truejacs), title="Jacobian entries", xlabel="Time", label="True Jacobian", c=:green, primary=primary, layout=4)
plot!(jacmat(normaljacs),  label="Normal training", c=:red, primary=primary)
plot!(jacmat(jacpropjacs),  label="Jacprop training", c=:blue, primary=primary)
# The tangent-space regularization restricts the flexibility of the model and promotes a model where the jacobian changes slowly along the trajectories in the training data.
