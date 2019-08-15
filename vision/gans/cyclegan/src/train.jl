using CUDAnative
device!(2)
using Images,CuArrays,Flux
using Flux:@treelike, Tracker
using Base.Iterators: partition
using Random
using Statistics
using Flux.Tracker:update!
using BSON: @save,@load
using Flux:testmode!
using Distributions:Normal,Uniform

include("utils.jl")
include("generator.jl")
include("discriminator.jl")

TRAIN_A_PATH = "../data/trainA/"
TRAIN_B_PATH = "../data/trainB/"
SAVE_PATH = "../weights/"

# Hyperparameters
NUM_EPOCHS = 200
BATCH_SIZE = 2
dis_lr = 0.0002f0
gen_lr = 0.0002f0
λ₁ = convert(Float32,100.0) # Cycle loss weight for dommain A
λ₂ = convert(Float32,100.0) # Cycle loss weight for domain B
λid = convert(Float32,0.5) # Identity loss weight - Set this to '0' if identity loss is not required
NUM_EXAMPLES = 1 # Temporary for experimentation
VERBOSE_FREQUENCY = 10 # Verbose output after every 2 epochs
SAVE_FREQUENCY = 200

# Data Loading
dataA = load_dataset(TRAIN_A_PATH,256)[:,:,:,1:NUM_EXAMPLES] 
dataB = load_dataset(TRAIN_B_PATH,256)[:,:,:,1:NUM_EXAMPLES]
mb_idxs = partition(1:size(dataA)[end], BATCH_SIZE)
train_A = [make_minibatch(dataA, i) for i in mb_idxs]
train_B = [make_minibatch(dataB, i) for i in mb_idxs]
println("Loaded Data")

# Define Optimizers
opt_gen = ADAM(gen_lr,(0.5,0.999))
opt_disc_A = ADAM(dis_lr,(0.5,0.999))
opt_disc_B = ADAM(dis_lr,(0.5,0.999))

# Define models
gen_A = UNet() |> gpu # Generator For A->B
gen_B = UNet() |> gpu # Generator For B->A
dis_A = Discriminator() |> gpu # Discriminator For Domain A
dis_B = Discriminator() |> gpu # Discriminator For Domain B
println("Loaded Models")

function dA_loss(a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    fake_A = gen_B(b) # Fake image generated in domain A
    fake_A_prob = dis_A(fake_A.data) # Probability that generated image in domain A is real
    real_A_prob = dis_A(a) # Probability that original image in domain A is real

    real_labels = ones(size(real_A_prob)...) |> gpu
    fake_labels = zeros(size(fake_A_prob)...) |> gpu

    dis_A_real_loss = ((real_A_prob .- real_labels).^2)
    dis_A_fake_loss = ((fake_A_prob .- fake_labels).^2)

    mean(dis_A_real_loss + dis_A_fake_loss)
end

function dB_loss(a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    fake_B = gen_A(a) # Fake image generated in domain B
    fake_B_prob = dis_B(fake_B.data) # Probability that generated image in domain B is real
    real_B_prob = dis_B(b) # Probability that original image in domain B is real

    real_labels = ones(size(real_B_prob)...) |> gpu
    fake_labels = zeros(size(fake_B_prob)...) |> gpu

    dis_B_real_loss = ((real_B_prob .- real_labels).^2)
    dis_B_fake_loss = ((fake_B_prob .- fake_labels).^2)

    mean(dis_B_real_loss + dis_B_fake_loss)
end

function g_loss(a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    # Forward Propogation # 
    fake_B = gen_A(a) # Fake image generated in domain B
    fake_B_prob = dis_B(fake_B) # Probability that generated image in domain B is real
    real_B_prob = dis_B(b) # Probability that original image in domain B is real

    fake_A = gen_B(b) # Fake image generated in domain A
    fake_A_prob = dis_A(fake_A) # Probability that generated image in domain A is real
    real_A_prob = dis_A(a) # Probability that original image in domain A is real
    
    real_labels = ones(size(real_B_prob)...) |> gpu
    fake_labels = zeros(size(fake_B_prob)...) |> gpu
        
    rec_A = gen_B(fake_B)
    rec_B = gen_A(fake_A)
    
    ### Generator Losses ###
    # For domain A->B  #
    gen_B_loss = mean((fake_B_prob .- real_labels).^2)
    rec_B_loss = mean(abs.(b .- rec_B)) # Reconstruction loss for domain B
    
    # For domain B->A  #
    gen_A_loss = mean((fake_A_prob .- real_labels).^2)
    rec_A_loss = mean(abs.(a .- rec_A)) # Reconstrucion loss for domain A

    # Identity losses 
    # gen_A should be identity if b is fed : ||gen_A(b) - b||
    idt_A_loss = mean(abs.(gen_A(a) .- b))
    # gen_B should be identity if a is fed : ||gen_B(a) - a||
    idt_B_loss = mean(abs.(gen_B(b) .- a))

    gen_A_loss + gen_B_loss + λ₁*rec_A_loss + λ₂*rec_B_loss + λid*(λ₁*idt_A_loss + λ₂*idt_B_loss
end

# Forward prop, backprop, optimise!
function train_step(X_A,X_B,opt_gen,opt_disc_A,opt_disc_B) 
    # Normalise the Images
    X_A = norm(X_A)
    X_B = norm(X_B)

    # Optimise Generators
    gs = Tracker.gradient(() -> g_loss(X_A,X_B),params(params(gen_A)...,params(gen_B)...))
        
    update!(opt_gen,params(params(gen_A)...,params(gen_B)...),gs)

    # Optimise Discriminators
    gs = Tracker.gradient(() -> dA_loss(X_A,X_B),params(dis_A))
    update!(opt_disc_A,params(dis_A),gs)

    gs = Tracker.gradient(() -> dB_loss(X_A,X_B),params(dis_B))
    update!(opt_disc_B,params(dis_B),gs)
end

function save_weights(gen_A,dis_A,gen_B,dis_B)
    gen_A = gen_A |> cpu
    gen_B = gen_B |> cpu
    dis_A = dis_A |> cpu
    dis_B = dis_B |> cpu
    @save string(SAVE_PATH,"gen_A.bson") gen_A
    @save string(SAVE_PATH,"gen_B.bson") gen_B
    @save string(SAVE_PATH,"dis_A.bson") dis_A
    @save string(SAVE_PATH,"dis_B.bson") dis_B
end

function train()
    println("Training...")
    verbose_step = 0
    for epoch in 1:NUM_EPOCHS
        println("-----------Epoch : $epoch-----------")
        for i in 1:length(train_A)
            g_loss,dA_loss,dB_loss = train_step(train_A[i] |> gpu,train_B[i] |> gpu,opt_gen,opt_disc_A,opt_disc_B)
        end
    	if epoch % SAVE_FREQUENCY == 0
    		save_weights(gen_A,dis_A,gen_B,dis_B)
    	end
    end
    save_weights(gen_A,dis_A,gen_B,dis_B)
    println("Saved...")
end

train()
