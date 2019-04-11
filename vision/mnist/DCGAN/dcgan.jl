# Get the imports done
using Flux, Flux.Data.MNIST
using Flux: @epochs, back!, testmode!, throttle
using Base.Iterators: partition
using Distributions: Uniform,Normal
using CUDAnative: tanh, log, exp
using CuArrays
using Images
using Statistics

# Define the hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 15
noise_dim = 100
channels = 128
hidden_dim = 7 * 7 * channels
training_steps = 0
verbose_freq = 100
dis_lr = 0.0001f0 # Discriminator Learning Rate
gen_lr = 0.0001f0 # Generator Learning Rate

# Loading Data

# We use Flux's built in MNIST Loader
imgs = MNIST.images()

# Partition into batches of size 'BATCH_SIZE'
data = [reshape(float(hcat(vec.(imgs)...)),28,28,1,:) for imgs in partition(imgs, BATCH_SIZE)]

# Define out distribution for random sampling for the generator to sample noise from
dist = Normal(0.0,1.0) # Standard Normal noise is found to give better results

expand_dims(x,n::Int) = reshape(x,ones(Int64,n)...,size(x)...)
squeeze(x) = dropdims(x, dims = tuple(findall(size(x) .== 1)...))

# The Generator
generator = Chain(
            Dense(noise_dim, 1024, leakyrelu), 
            x->expand_dims(x,1),
            BatchNorm(1024),
            x->squeeze(x),
            Dense(1024, hidden_dim, leakyrelu), 
            x->expand_dims(x,1),
            BatchNorm(hidden_dim),
            x->squeeze(x),
            x->reshape(x,7,7,channels,:),
            ConvTranspose((4,4), channels=>64, relu; stride=(2,2), pad=(1,1)), 
            x->expand_dims(x,2),
            BatchNorm(64),
            x->squeeze(x),
            ConvTranspose((4,4), 64=>1, tanh; stride=(2,2), pad=(1,1))
            ) |> gpu

# The Discriminator
discriminator = Chain(
                Conv((3,3), 1=>32, leakyrelu;pad = 1), 
                x->meanpool(x, (2,2)),
                Conv((3,3), 32=>64, leakyrelu;pad = 1), 
                x->meanpool(x, (2,2)),
                x->reshape(x,7*7*64,:),
                Dense(7*7*64, 1024, leakyrelu), 
                x->expand_dims(x,1),
                BatchNorm(1024),
                x->squeeze(x),
                Dense(1024, 1,sigmoid)
                ) |> gpu

# Define the optimizers

opt_gen  = ADAM(params(generator),gen_lr, β1 = 0.5)
opt_disc = ADAM(params(discriminator),dis_lr, β1 = 0.5)

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

img(x) = Gray.(reshape((x+1)/2, 28, 28)) # For denormalizing the generated image

function sample()
  noise = [rand(dist, noise_dim, 1) for i=1:9] # Sample 9 digits
  noise = gpu.(noise) # Add to GPU
  
  testmode!(generator)
  fake_imgs = img.(map(x -> gpu(generator(x).data), noise)) # Generate a new image from random noise
  testmode!(generator, false)
  
  img_grid = vcat([hcat(imgs...) for imgs in partition(fake_imgs, 3)]...) # Create grid for saving
end

cd(@__DIR__)


# We use the Binary Cross Entropy Loss
function bce(ŷ, y)
    mean(-y.*log.(ŷ) - (1  .- y .+ 1f-10).*log.(1 .- ŷ .+ 1f-10))
end

function train(x)
  global training_steps
  println("TRAINING")
  z = rand(dist, noise_dim, BATCH_SIZE) |> gpu
  inp = 2x .- 1 |> gpu # Normalize images to [-1,1]
 
  zero_grad!(discriminator)
  
  D_real = discriminator(inp) # D(x)
  real_labels = ones(size(D_real)) |> gpu
    
  
  D_real_loss = bce(D_real,real_labels)

  fake_x = generator(z) # G(z)
  D_fake = discriminator(fake_x) # D(G(z))
  fake_labels = zeros(size(D_fake)) |> gpu  
    
  D_fake_loss = bce(D_fake,fake_labels)

  D_loss = D_real_loss + D_fake_loss
  Flux.back!(D_loss)
  opt_disc() # Optimize the discriminator

  zero_grad!(generator)

  fake_x = generator(z) # G(z)
  D_fake = discriminator(fake_x) # D(G(z))
  real_labels = ones(size(D_fake)) |> gpu  

  G_loss = bce(D_fake,real_labels)

  Flux.back!(G_loss)
  opt_gen() # Optimise the generator
  
  if training_steps % verbose_freq == 0
    println("D Loss: $(D_loss.data) | G loss: $(G_loss.data)")
  end

  training_steps += 1
end

for e = 1:NUM_EPOCHS
  for imgs in data
    train(imgs)
  end
  println("Epoch $e over.")
end

save("sample_dcgan.png", sample())
