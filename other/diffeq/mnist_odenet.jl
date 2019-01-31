using Flux, DiffEqFlux, DifferentialEquations, Plots, StatsBase, RecursiveArrayTools
using Flux: onehotbatch
using MLDatasets:MNIST
using Base.Iterators: repeated,partition

batch_size=32

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
               saveat=batch_t, reltol=1e-3,abstol=1e-3)
end

model = Chain(downsample,u->n_ode(Flux.data(u),25.)[:,:,:,:,1],classify)

loss(x,y) = Flux.mse(model(x),y)

opt = ADAM(0.1)

Ps = params(model)

Flux.train!(loss, Ps, train_data[1:2], opt, cb = ())
