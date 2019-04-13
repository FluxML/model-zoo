# Get the imports done
using Flux, Flux.Data.MNIST,Flux
using Flux: @epochs, back!, testmode!, throttle, Tracker
using Flux.Tracker:update!
using Base.Iterators: partition,flatten
using Flux: onehot,onehotbatch
using Distributions: Normal
using Statistics
using Images

# Define the hyperparameters
NUM_EPOCHS = 500
BATCH_SIZE = 100
NOISE_DIM = 100
gen_lr = 0.0001f0 # Generator learning rate
dis_lr = 0.0001f0 # discriminator learning rate
training_steps = 0
verbose_freq = 2

# Loading Data
@info("Loading data set")
train_labels = MNIST.labels()[1:100] |> gpu
train_imgs = MNIST.images()[1:100] |> gpu

# Bundle images together with labels and group into minibatches
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, 784, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, i] = Float32.(reshape(X[idxs[i]],784))
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return vcat(X_batch, Y_batch)
end

mb_idxs = partition(1:length(train_imgs), BATCH_SIZE)
train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs]

# Define out distribution for random sampling for the generator to sample noise from
dist = Normal(0.0,1.0) # Standard Normal noise is found to give better results

# The Generator
generator = Chain(Dense(NOISE_DIM + 10,1200,leakyrelu),
                  Dense(1200,1000,leakyrelu),
                  Dense(1000,784,tanh)
                  ) |> gpu

# The Discriminator
discriminator = Chain(Dense(794,512,leakyrelu),
                      Dense(512,128,leakyrelu),
                      Dense(128,1,sigmoid)
                      ) |> gpu

# Define the optimizers
opt_gen  = ADAM(gen_lr)
opt_disc = ADAM(dis_lr)

# Utility functions to zero out our model gradients
function nullify_grad!(p)
  if typeof(p) <: TrackedArray
    p.grad .= 0.0f0
  end
  return p
end

function zero_grad!(model)
  model = mapleaves(nullify_grad!, model)
end

# Creating and Saving Utilities

img(x) = Gray.(reshape((x.+1)/2, 28, 28, 1)) # For denormalizing the generated image

function sample()
  num_samples = 9 # Number of digits to sample
  fake_labels = zeros(10,num_samples)
  for i in 1:num_samples
        fake_labels[rand(1:9),i] = 1
  end

  noise = [vcat(rand(dist, NOISE_DIM, 1),fake_labels[:,i]) for i=1:num_samples] # Sample 9 digits
  noise = gpu.(noise) # Add to GPU

  testmode!(generator)
  fake_imgs = img.(map(x -> gpu(generator(x).data), noise)) # Generate a new image from random noise
  testmode!(generator, false)

  img_grid = fake_imgs[1]
end

cd(@__DIR__)

# We use the Binary Cross Entropy Loss
function bce(ŷ, y)
    mean(-y.*log.(ŷ .+ 1f-10) - (1  .- y .+ 1f-10).*log.(1 .- ŷ .+ 1f-10))
end

function train(x)
  global training_steps

  z = rand(dist,NOISE_DIM, BATCH_SIZE) |> gpu
  inp = 2x .- 1 |> gpu # Normalize images to [-1,1]
  inp[end-9:end,:] = x[end-9:end,:] # The labels should not be modified

  labels = Float32.(x[end-9:end,:]) |> gpu # y
  zero_grad!(discriminator)
  zero_grad!(generator)

  D_real = discriminator(inp) # D(x|y)
  real_labels = ones(size(D_real)) |> gpu

  D_real_loss = bce(D_real,real_labels)

  fake_x = generator(vcat(z,labels)) # G(z|y)
  D_fake = discriminator(vcat(fake_x,labels)) # D(G(z|y))
  fake_labels = zeros(size(D_fake)) |> gpu  

  D_fake_loss = bce(D_fake,fake_labels)

  D_loss = D_real_loss + D_fake_loss
  # Flux.back!(D_loss)
  # opt_disc() # Optimize the discriminator
  gs = Tracker.gradient(() -> D_loss,params(discriminator))
  update!(opt_disc,params(discriminator),gs)

  zero_grad!(discriminator)
  zero_grad!(generator)

  fake_x = generator(vcat(z,labels)) # G(z|y)
  D_fake = discriminator(vcat(fake_x,labels)) # D(G(z|y))
  real_labels = ones(size(D_fake)) |> gpu  

  G_loss = bce(D_fake,real_labels)
  # Flux.back!(G_loss)
  # opt_gen() # Optimise the generator
  gs = Tracker.gradient(() -> G_loss,params(generator))
  for _ in 1:3
	  update!(opt_gen,params(generator),gs)
  end

  if training_steps % verbose_freq == 0
    println("D Loss: $(D_loss.data) | G loss: $(G_loss.data)")
  end

  println(training_steps)
  training_steps += 1
end

for e = 1:NUM_EPOCHS
  for data in train_set
    train(data)
  end
  println("Epoch $e over.")
end

save("sample_cgan.png", sample()) 