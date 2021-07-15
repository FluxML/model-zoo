using CUDAnative
using CUDAnative:exp,log
device!(3)
println("Device Selected")

using Images,CuArrays,Flux
using Metalhead
using Flux:@treelike, Tracker
using Base.Iterators: partition
using Random
using Statistics
using Flux.Tracker:update!
using BSON: @save
using Flux:testmode!
using Distributions:Normal,Uniform
using JLD

# HYPER PARAMETERS
UP_SAMPLE_FACTOR = 4
UP_SAMPLE_FACTOR_STEP = 2
NUM_EPOCHS = 200
BATCH_SIZE = 1
dis_lr = 0.0002f0
gen_lr = 0.0002f0
NUM_EXAMPLES = 1  # Temporary for experimentation
VERBOSE_FREQUENCY = 1 # Verbose output after every 10 steps
SAVE_FREQUENCY = 500
SAMPLE_FREQUENCY = 5 # Sample every these mamy number of steps
# Debugging
G_STEPS = 1
D_STEPS = 1
resume = false
# Image Size 
H = 1404
W = 2040
RESIZE_FACTOR = 1
# Generator Number of Blocks
B = 8

include("utils.jl")
include("layers.jl")
include("generator.jl")
include("discriminator.jl")

BASE_PATH = "../../../../../../../references/srgan/"
HR_PATH = string(BASE_PATH,"DIV2K_train_HR/")
LR_PATH = string(BASE_PATH,"DIV2K_train_LR_bicubic/X4/")

img_HR,img_LR = load_dataset(HR_PATH,LR_PATH)
img_HR = img_HR[1:NUM_EXAMPLES]
img_LR = img_LR[1:NUM_EXAMPLES]
mb_idxs = partition(shuffle!(collect(1:length(img_HR))), BATCH_SIZE)
train_HR_batches = [img_HR[i] for i in mb_idxs]
train_LR_batches = [img_LR[i] for i in mb_idxs]

# Optimizers
opt_gen = ADAM(gen_lr,(0.5,0.999))
opt_disc = ADAM(dis_lr,(0.5,0.999))

# Load VGG net
# vgg = VGG19() |> gpu
# vgg = Chain(vgg.layers[1:20]...)
# println("Loaded VGG net")

gen = Gen(B) |> gpu
println("Loaded Generator")
dis = Discriminator() |> gpu
println("Loaded Discriminator")

function d_loss(X_HR,X_LR)
    X_SR = gen(X_LR).data # Super-resolution image
    fake_prob = dis(X_SR)
    fake_labels = zeros(size(fake_prob)...) |> gpu
    loss_D_fake = bce(fake_prob,fake_labels)
	
    real_prob = dis(X_HR)
    real_labels = ones(size(real_prob)...) |> gpu
    loss_D_real = bce(real_prob,real_labels)

    mean(loss_D_real .+ loss_D_fake)
end

function g_loss(X_HR,X_LR)
	# Adversarial loss
    X_SR = gen(X_LR)
    fake_prob = dis(X_SR)
    real_labels = ones(size(fake_prob)...) |> gpu
    loss_adv = mean(bce(fake_prob,real_labels))

    # HR_features = vgg(X_HR).data
    # SR_features = vgg(X_SR)
    # content_loss = mean((HR_features .- SR_features).^2)

    loss_adv # + 0.001f0 * content_loss
end

function train_step(X_HR,X_LR)
	X_HR = norm(X_HR)

	for _ in 1:D_STEPS
   	   	gs = Tracker.gradient(() -> d_loss(X_HR,X_LR),params(dis))
   	   	update!(opt_disc,params(dis),gs)
    end
    
    for _ in 1:G_STEPS
	    gs = Tracker.gradient(() -> g_loss(X_HR,X_LR),params(gen))  
	    update!(opt_gen,params(gen),gs)
	end
end

function train()
	println("Training...")
    verbose_step = 0

    for epoch in 1:NUM_EPOCHS
        println("-----------Epoch : $epoch-----------")

        for i in 1:length(train_HR_batches)
			X_HR,X_LR = get_batch(train_HR_batches[i],train_LR_batches[i],H,W)

            train_step(X_HR |> gpu,X_LR |> gpu)
        end
    end
end

train()
