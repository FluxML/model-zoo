using Flux, DiffEqFlux, DifferentialEquations, Plots

## Setup DDE to optimize
function delay_lotka_volterra(du,u,h,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)*h(p,t-0.1)[1]
  du[2] = dy = (δ*x - γ)*y
end

h(p,t) = ones(eltype(p),2)
prob = DDEProblem(delay_lotka_volterra,[1.0,1.0],h,(0.0,10.0),constant_lags=[0.1])
p = param([2.2, 1.0, 2.0, 0.4])
params = Flux.Params([p])
function predict_rd_dde()
  diffeq_rd(p,prob,MethodOfSteps(Tsit5()),saveat=0.1)[1,:]
end
loss_rd_dde() = sum(abs2,x-1 for x in predict_rd_dde())

# Optimize the parameters so the DDE's solution stays near 1

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  display(loss_rd_dde())
  # using `remake` to re-create our `prob` with current parameters `p`
  display(plot(solve(remake(prob,p=Flux.data(p)),MethodOfSteps(Tsit5()),saveat=0.1),ylim=(0,6)))
end
# Display the DDE with the initial parameter values.
cb()
Flux.train!(loss_rd_dde, params, data, opt, cb = cb)
