using CUDAnative
using CUDAnative:exp,log
device!(3)
println("Device Selected")

using Images,CuArrays,Flux
using Flux:@treelike, Tracker
using Base.Iterators: partition
using Random
using Statistics
using Flux.Tracker:update!
using BSON: @save
using Flux:testmode!
using Distributions:Normal,Uniform
using JLD

include("utils.jl")
include("generator.jl")
include("discriminator.jl")
include("test.jl")

# Hyperparameters
NUM_EPOCHS = 2000
BATCH_SIZE = 4
dis_lr = 0.0002f0
gen_lr = 0.0002f0
λ = 100.0f0 # L1 reconstruction Loss Weight
NUM_EXAMPLES = 10  # Temporary for experimentation
VERBOSE_FREQUENCY = 1 # Verbose output after every 10 steps
SAVE_FREQUENCY = 500
SAMPLE_FREQUENCY = 5 # Sample every these mamy number of steps
# Debugging
G_STEPS = 1 
D_STEPS = 1
resume = false
inverse_order = true
# Define paths
DATA_PATH = "../../../../../../old_server_pix2pix/data/facades/train/"
SAVE_PATH = "../weights/"

# Global printing variables
global gloss = 0.0
global dloss = 0.0

# Statistics to keep track of
global gloss_hist = []
global dloss_hist = []
global global_step = 0

# Data Loading
data = load_dataset(DATA_PATH,256)[1:NUM_EXAMPLES]

mb_idxs = partition(shuffle!(collect(1:length(data))), BATCH_SIZE)
train_batches = [data[i] for i in mb_idxs]
println("Loaded Data")

function d_loss(gen,dis,a,b)
	"""
	a : Image in domain A
	b : Image in domain B
	"""
	fake_B = gen(a).data

	fake_AB = cat(fake_B,a,dims=3)

	fake_prob = dis(fake_AB)

	fake_labels = zeros(size(fake_prob)...) |> gpu
	loss_D_fake = mean(bce(fake_prob,fake_labels))

	real_AB =  cat(b,a,dims=3)
	real_prob = dis(real_AB)
	real_labels = ones(size(real_prob)...) |> gpu

	loss_D_real = mean(bce(real_prob,real_labels))

	return loss_D_real + loss_D_fake
end

function g_loss(gen,dis,a,b)
	"""
	a : Image in domain A
	b : Image in domain B
	"""
	fake_B = gen(a)
	fake_AB = cat(fake_B,a,dims=3)

	fake_prob = dis(fake_AB)
	
	real_labels = ones(size(fake_prob)...) |> gpu

	loss_adv = mean(bce(fake_prob,real_labels))

	loss_L1 = mean(abs.(fake_B .- b))

	return loss_adv + λ * loss_L1
end

# Forward prop, backprop, optimise!
function train_step(gen,dis,X_A,X_B,opt_gen,opt_disc)
	X_A = norm(X_A)
	X_B = norm(X_B)

	for _ in 1:D_STEPS
	   gs = Tracker.gradient(() -> d_loss(gen,dis,X_A,X_B),params(dis))   
	   update!(opt_disc,params(dis),gs)
	 end

	for _ in 1:G_STEPS
		gs = Tracker.gradient(() -> g_loss(gen,dis,X_A,X_B),params(gen))  
		update!(opt_gen,params(gen),gs)
	end
end

function save_weights(gen,dis)
	gen = gen |> cpu
	dis = dis |> cpu
	@save string(SAVE_PATH,"gen.bson") gen
	@save string(SAVE_PATH,"dis.bson") dis
	gen = gen |> gpu
	dis = dis |> gpu
	println("Saved...")
end

function train()
	global gloss
	global dloss
	global global_step

	println("Training...")
	verbose_step = 0

	# Define models
	if resume == true
		@load string(SAVE_PATH,"gen.bson") gen
		@load string(SAVE_PATH,"dis.bson") dis
		gen = gen |> gpu
		dis = dis |> gpu
		println("Loaded Networks")
	else
		gen = UNet() |> gpu # Generator For A->B
		dis = Discriminator() |> gpu
		println("Initialized Neworks")
	end

	opt_gen = ADAM(gen_lr,(0.5,0.999))
	opt_disc = ADAM(dis_lr,(0.5,0.999))

	for epoch in 1:NUM_EPOCHS
		println("-----------Epoch : $epoch-----------")
	
		mb_idxs = partition(shuffle!(collect(1:length(data))), BATCH_SIZE)
		train_batches = [data[i] for i in mb_idxs]
	
		for i in 1:length(train_batches)
		    global_step += 1
			println(i)
			
		    if global_step % 7000 == 0
			  	opt_gen.eta = opt_gen.eta / 2.0
			  	opt_disc.eta = opt_disc.eta / 2.0
		    end

			if inverse_order == false
			   train_A,train_B = get_batch(train_batches[i],256)
			else
			   train_B,train_A = get_batch(train_batches[i],256)
			end

			train_step(gen,dis,train_A |> gpu,train_B |> gpu,opt_gen,opt_disc)
			if verbose_step % SAMPLE_FREQUENCY == 0	
				sampleA2B(train_A,gen)
			end
			
			if verbose_step % SAVE_FREQUENCY == 0
				save_weights(gen,dis)
			end
		end
	end

	save_weights(gen,dis)
end

train()
