using Flux, Flux.Data.MNIST
using Flux: @epochs, back!, testmode!
using Base.Iterators: partition
using Distributions: Uniform
using CuArrays

# Encode MNIST images as compressed vectors that can later be decoded back into
# images.
BATCH_SIZE = 128

imgs = MNIST.images()

# Partition into batches of size 128
data = [float(hcat(vec.(imgs)...)) for imgs in partition(imgs, BATCH_SIZE)]
#data = gpu.(data)

NUM_EPOCHS = 50
noise_dim = 96
training_steps = 0
VERBOSE_FREQ = 100
dist = Uniform(-1, 1)

############################### WGAN Architecture ##############################
################################## Generator ###################################

generator = Chain(Dense(noise_dim, 1024, relu), Dense(1024, 1024, relu), Dense(1024, 784, tanh)) |> gpu

################################## Discriminator ###############################

discriminator = Chain(Dense(784, 256, leakyrelu), Dense(256, 256, leakyrelu), Dense(256, 1)) |> gpu

################################################################################

opt_gen  = ADAM(params(generator), 0.001f0, β1 = 0.5)
opt_disc = ADAM(params(discriminator), 0.001f0, β1 = 0.5)

############################### Helper Functions ###############################

function nullify_grad!(p)
  if typeof(p) <: TrackedArray
    p.grad .= 0.0f0
  end
  return p
end

function zero_grad!(model)
  model = mapleaves(nullify_grad!, model)
end

############################# Generating Sample Images #########################
using Images

img(x) = Gray.(reshape((x+1)/2, 28, 28))

function sample()
  # 36 random digits
  noise = [rand(dist, noise_dim, 1) |> gpu for i=1:36]

  # generating images
  testmode!(generator)
  fake_imgs = img.(map(x -> cpu(generator(x).data), noise))
  testmode!(generator, false)
  
  # Stack them all together
  vcat([hcat(imgs...) for imgs in partition(fake_imgs, 6)]...)
end

cd(@__DIR__)

################################################################################

function train(x)
  global training_steps
  
  z = rand(dist, noise_dim, BATCH_SIZE) |> gpu
  inp = 2x - 1 |> gpu
 
  zero_grad!(discriminator)
   
  D_real = discriminator(inp)
  D_real_loss = mean((D_real - 1.0f0).^2 / 2.0f0)

  fake_x = generator(z)
  D_fake = discriminator(fake_x)
  D_fake_loss = mean(D_fake.^2 / 2.0f0)

  D_loss = D_real_loss + D_fake_loss

  back!(D_loss)
  opt_disc()

  zero_grad!(generator)
  z = rand(dist, noise_dim, BATCH_SIZE) |> gpu
  fake_x = generator(z)
  D_fake = discriminator(fake_x)
 
  G_loss = mean((D_fake - 1.0f0).^2 / 2.0f0)

  back!(G_loss)
  opt_gen()

  if training_steps % VERBOSE_FREQ == 0
    println("D loss: $(D_loss.data) | G loss: $(G_loss.data)")
  end

  #param(0.0f0)
end

for e=1:NUM_EPOCHS
  for imgs in data
    train(imgs)
  end
  println("EPOCH $e Over")
end

save("sample_lsgan.png", sample())
