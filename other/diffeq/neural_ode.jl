using Flux, DiffEqFlux, DifferentialEquations, Plots, StatsBase

const u0 = Float32[2.; 0.]
const datasize = 30
const batchsize = 30
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
    batch_idxs = sample(1:datasize,batchsize,
                        replace=false,ordered=true) 
    batch_t = true_sol.t[batch_idxs]
    batch_u = Array(true_sol[batch_idxs])
    #= batch_u0 = batch_u[1] =#
    #= return batch_u0, batch_u, batch_t =#
end

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
  display(loss_n_ode())
  # plot current prediction against data
  cur_pred = Flux.data(predict_n_ode())
  pl = scatter(t,ode_data[1,:],label="data")
  scatter!(pl,t,cur_pred[1,:],label="prediction")
  display(plot(pl))
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)
