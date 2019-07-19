using Flux, DiffEqFlux, OrdinaryDiffEq, StatsBase, RecursiveArrayTools
using Distributions

const tspan = (0.0,25)
const RANGE = (Float32(-8.),Float32(8.))
const BS = 200

target(u) = u.^3

function gen_data(batchsize,target_fun)
    x = Float32.(rand(Uniform(RANGE...,),batchsize))'|>collect
    return x,target(x)
end
data = Iterators.repeated(gen_data(BS,target), 10000) 
u0_test,y0_test= first(data)

dudt = Chain(Dense(1,20,tanh),Dense(20,1))

function n_ode(u0)
    neural_ode(dudt,u0,tspan,Tsit5(),
               save_start=false,
               save_everystep=false,
               reltol=1e-5,abstol=1e-12)
end
ps = Flux.params(dudt)
n_ode(u0_test)

loss(x,y) = mean(abs.(n_ode(x).-y))

opt = ADAM(0.1)
cb = function () #callback function to observe training
    tx,ty = gen_data(BS,target)
    display(loss(tx,ty))
end

n_ode(u0_test)

g = Tracker.gradient(()->loss(u0_test,y0_test),ps)

g[dudt[1].W]

[print(g) for g in Tracker.gradient(()->loss(u0_test,y0_test),ps)] # no gradients


# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss, ps, data, opt, cb = cb)
