using DiffEqFlux, Flux, Test, OrdinaryDiffEq
using Statistics
#= using Plots =#

## True Solution
u0 = [2.; 0.]
datasize = 30
tspan = (0.0,25.0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

true_prob = ODEProblem(trueODEfunc, u0,tspan)
true_sol = solve(true_prob,Tsit5(),saveat=range(tspan[1],tspan[2],length=datasize))

#= true_sol_plot = solve(true_prob,Tsit5()) =#
#= plot(true_sol_plot) =#

## Neural ODE
dudt = Chain(Dense(2,50,tanh),Dense(50,2))

function ODEfunc(du,u,p,t)
    du .= Flux.data(dudt(u))
end

pred_prob = ODEProblem(ODEfunc, u0,tspan)
pred_sol = solve(pred_prob,Tsit5(),saveat=range(tspan[1],tspan[2],length=datasize))

## Loss
l1_loss(pred,target) = mean(abs.(pred-target))
l1_loss(pred_sol,true_sol)
