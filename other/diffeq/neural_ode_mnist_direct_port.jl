using OrdinaryDiffEq
using Flux
using Flux: logitcrossentropy
using MLDatasets: MNIST
using MLDataUtils
using DiffEqFlux
using CuArrays; CuArrays.allowscalar(false)
using NNlib
using IterTools: ncycle

function loadmnist(batchsize=bs)
	# Use MLDataUtils LabelEnc for natural onehot conversion
    onehot(labels_raw) = convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:9)))
	# Load MNIST
	imgs, labels_raw = MNIST.traindata();
	# Process images into (H,W,C,BS) batches
	x_train = reshape(imgs,size(imgs,1),size(imgs,2),1,size(imgs,3))|>gpu
	x_train = batchview(x_train,batchsize);
	# Onehot and batch the labels
	y_train = onehot(labels_raw)|>gpu
	y_train = batchview(y_train,batchsize)
	return x_train, y_train
end

# Main
const bs = 128
x_train, y_train = loadmnist(bs);

down = Chain(
             Conv((3,3),1=>64,relu,stride=1), GroupNorm(64,64),
             Conv((4,4),64=>64,relu,stride=2,pad=1), GroupNorm(64,64),
             Conv((4,4),64=>64,stride=2,pad=1),
            )|>gpu;
dudt = Chain(
           Conv((3,3),64=>64,relu,stride=1,pad=1),
           Conv((3,3),64=>64,relu,stride=1,pad=1)
          ) |>gpu;
fc = Chain(GroupNorm(64,64),
           x->relu.(x),
           MeanPool((6,6)),
           x -> reshape(x,(64,bs)),
           Dense(64,10)
          )|>gpu;

solver_kwargs = Dict(:save_start=>false,:save_everystep=>false, :reltol=>1e-3, :abstol=>1e-3)
neural_ode_layer = DiffEqFlux.NeuralODE(dudt, (0.f0,1.f0), Tsit5(), solver_kwargs)

model = Chain(
              down,             #(28,28,1,BS) -> (6,6,64,BS)
              neural_ode_layer, #(6,6,64,BS) -> (6,6,64,BS)
              fc                #(6,6,64,BS) -> (10, BS)
             )

# Showing this works with forward solve
x_m = model(x_train[1])

# Define loss and test with forward solve
function loss(x,y)
    y_hat = model(x)
    return logitcrossentropy(y_hat,y)
end

loss(x_train[1],y_train[1])

# Define accuracy, diagnostic but not objective
classify(x) = argmax.(eachcol(x))

function accuracy(model,data; n_batches=100)
    total_correct = 0
    total = 0
    for (x,y) in collect(data)[1:n_batches]
        target_class = classify(cpu(y))
        predicted_class = classify(cpu(model(x)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct/total
end

accuracy(model, zip(x_train,y_train))

# Logging and accuracy in callback
using TensorBoardLogger, Logging

lg=TBLogger("tensorboard_logs/run", min_level=Logging.Info)
iter = 0
cb() = begin
    global iter += 1
    li = Flux.data(loss(x_train[1],y_train[1]))
    with_logger(lg) do
        @info "loss" li
    end
    if iter%100 == 0
        ai = accuracy(model, zip(x_train,y_train))
        with_logger(lg) do
            @info "accuracy" ai
        end
        ai > 0.97 && Flux.stop()
    end
end

#Main Train Loop
opt=ADAM()
Flux.train!(loss,params(model),ncycle(zip(x_train,y_train),100),opt, cb=cb)

# Save and load trained models for plotting
using BSON: @save, @load

#model_cpu = cpu(model)
#@save "saved_models/mnist_node.bson" model_cpu

@load "saved_models/mnist_node.bson" model_cpu

dwn_loaded = gpu(model_cpu[1])
node_loaded= gpu(model_cpu[2])

dense_solver_kwargs = Dict(:saveat=> collect(0.f0:0.01:1.f0),:reltol=>1e-3, :abstol=>1e-3)
dense_node = DiffEqFlux.NeuralODE(node_loaded.model,node_loaded.tspan,node_loaded.solver,node_loaded.args,dense_solver_kwargs)

# Animation code for JuliaCon talk: 
slv = cpu(Flux.data(dense_node(dwn_loaded(x_train[1]))))

using Plots; pyplot();

cs = palette(:default)

clfy(x) = argmax.(eachcol(x))

for ti in 1:size(slv,5) 
    plot(
         collect(eachrow(slv[2,3,7,:,1:ti])),
         collect(eachrow(slv[3,3,7,:,1:ti])), 
         legend=false,
         ticks=false,
         c=cs[clfy(y_train[1])]',
        opacity=0.5,
        xlim= extrema(slv[2,3,7,:,1:end]),
        ylim = slv[3,3,7,:,1:end])
    scatter!(
         collect(eachrow(slv[2,3,7,:,ti])),
         collect(eachrow(slv[3,3,7,:,ti])), 
         legend=false,
         ticks=false,
         c=cs[clfy(y_train[1])]')
    savefig("plots/mnist_dynamics_anim/$(lpad(ti,4,"0")).png")
end
