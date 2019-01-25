using Flux, DiffEqFlux, DifferentialEquations, Plots, StatsBase, RecursiveArrayTools

const u0 = Float32[2.; 0.]
const datasize = 1000
const batchsize = 30
const batchtime = 10
const tspan = (0.0f0,25f0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
true_sol = solve(prob,Tsit5(),saveat=t)
ode_data = Array(true_sol)

function get_batch()
    batch_0s = sample(1:datasize-batchtime, batchsize, replace=false,ordered=false)
    batch_idxs = range.(batch_0s,batch_0s.+(batchtime-1))
    batch_ts = [t[i] for i in batch_idxs]
    batch_u0 = [true_sol[u0] for u0 in batch_0s]
    batch_u = [true_sol[:,idxs] for idxs in batch_idxs]
    return batch_u0, batch_u, batch_ts
end
batch_u0, batch_u, batch_ts = get_batch();


dudt = Chain(x -> x.^3,
             Dense(2,50,tanh),
             Dense(50,2))
ps = Flux.params(dudt)
function n_ode(batch_u0, batch_t)
    neural_ode(dudt,batch_u0,tspan,Tsit5(),
               saveat=batch_t, reltol=1e-7,abstol=1e-9)
end
#= n_ode = x->neural_ode(dudt,x,tspan,Tsit5(), =#
                      #= saveat=t,reltol=1e-7,abstol=1e-9) =#

n_ode(batch_u0, batch_ts)

pred = n_ode(u0,t) # Get the prediction using the correct initial condition
scatter(t,ode_data[1,:],label="data")
scatter!(t,Flux.data(pred[1,:]),label="prediction")

function predict_n_ode(batch_u0,batch_t)
  n_ode(batch_u0, batch_t)
end
loss_n_ode(batch_u0, batch_u,batch_t) = sum(abs2,batch_u .- predict_n_ode(batch_u0,batch_t))

data = Iterators.repeated(get_batch(), 1000)
opt = ADAM(0.1)
cb = function () #callback function to observe training
    bu0,bu,bt = get_batch()
    display(loss_n_ode(bu0,bu,bt))
  # plot current prediction against data
  #= cur_pred = Flux.data(predict_n_ode(bu0,bt)) =#
  #= pl = scatter(bt,ode_data[1,:],label="data") =#
  #= scatter!(pl,bt,cur_pred[1,:],label="prediction") =#
  #= display(plot(pl)) =#
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)
