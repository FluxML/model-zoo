using Flux, DiffEqFlux, OrdinaryDiffEq,  StatsBase, RecursiveArrayTools
using Flux: onehotbatch
using MLDatasets:MNIST
using Base.Iterators: repeated,partition

batch_size=10

train_x, train_y = MNIST.traindata();
test_x, test_y = MNIST.testdata();
train_y_hot = onehotbatch(train_y,0:9);

train_data = [(reshape(train_x[:,:,i],(28,28,1,batch_size)),train_y_hot[:,i]) for i in partition(1:60_000,batch_size)];

downsample = Chain(
                   Conv((3,3),1=>32,stride=1),
                   BatchNorm(32,relu),
                   Conv((4,4),32=>32,stride=2,pad=1),
                   BatchNorm(32,relu),
                   Conv((4,4),32=>32,stride=2,pad=1)
                  )
dudt = Chain(
          BatchNorm(32,relu),
          Conv((3,3),32=>32,stride =1, pad=1),
          BatchNorm(32),
          Conv((3,3),32=>32,stride =1, pad=1),
          BatchNorm(32)
          )
classify = Chain(
                 BatchNorm(32,relu), 
                 MeanPool((6,6)),
                 x->view(x,1,1,:,:), #Need More General Flatten Here
                 Dense(32,10)
                )

ps = Flux.params(dudt)

function n_ode(batch_u0, batch_t)
    neural_ode(dudt,batch_u0,(0.,25.),Tsit5(),
               save_start=false,
               saveat=batch_t,      # Ugly way to only get sol[end] 
               reltol=1e-3,abstol=1e-3)
end

model = Chain(downsample,u->n_ode(u,[25.])[:,:,:,:,end],classify) # Further ugliness getting sol[end]

loss(x,y) = Flux.mse(model(x),y)

loss(train_data[1]...) # Works and is tracked
# This fails with neural_ode_rd with T undefined error

opt = ADAM(0.1)
Ps = params(model,ps)

Tracker.gradient(()->loss(xt,yt),Ps) # Fails with BoundsError
#ERROR: BoundsError: attempt to access 6×6×32×10×1 Array{Float32,5} at index [Base.Slice(Base.OneTo(6)), 1]

Flux.train!(loss, Ps, train_data, opt, cb = ()) # Fails with BoundsError


function ODEfunc(du,u,p,t)
    du.= dudt(u)
end
prob = ODEProblem(ODEfunc,downsample(xt),(0.,25.))

solve(prob,Tsit5(),saveat=25.)
