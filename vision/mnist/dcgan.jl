using Flux, Flux.Data.MNIST
using Flux: @epochs, back!, testmode!, throttle
using Base.Iterators: partition
using Distributions: Uniform
using CUDAnative: tanh, log, exp
using CuArrays

# Encode MNIST images as compressed vectors that can later be decoded back into
# images.
BATCH_SIZE = 128

imgs = MNIST.images()

# Partition into batches of size 128
data = [float(hcat(vec.(imgs)...)) for imgs in partition(imgs, BATCH_SIZE)]
#data = gpu.(data)

NUM_EPOCHS = 5
noise_dim = 96
channels = 128
hidden_dim = 7 * 7 * channels

dist = Uniform(-1, 1)

training_steps = 0
verbose_freq = 100
############################### DCGAN Architecture #############################
################################## Generator ###################################

fc_gen = Chain(Dense(noise_dim, 1024, relu), BatchNorm(1024),
            Dense(1024, hidden_dim, relu), BatchNorm(hidden_dim))
deconv_ = Chain(ConvTranspose((4,4), channels=>64, relu; stride=(2,2), pad=(1,1)), BatchNorm(64),
                ConvTranspose((4,4), 64=>1, tanh; stride=(2,2), pad=(1,1)))

generator = Chain(fc_gen..., x->reshape(x, 7, 7, channels, :), deconv_..., 
			     x->reshape(x, 784, :)) |> gpu

################################## Discriminator ###############################

fc_disc = Chain(Dense(1024, 1024, leakyrelu), Dense(1024, 1))
conv_ = Chain(Conv((5,5), 1=>32, leakyrelu), x->maxpool(x, (2,2)),
              Conv((5,5), 32=>64, leakyrelu), x->maxpool(x, (2,2)))

discriminator = Chain(x->reshape(x, 28, 28, 1, :), 
		      conv_..., x->reshape(x, 1024, :), fc_disc...) |> gpu

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

######################### Creating and saving the Images #######################

using Images

img(x) = Gray.(reshape((x+1)/2, 28, 28))

function sample()
  # 36 random digits
  noise = [rand(dist, noise_dim, 1) for i=1:36]
  noise = gpu.(noise)
  
  # Generating images
  testmode!(generator)
  fake_imgs = img.(map(x -> cpu(generator(x).data), noise))
  testmode!(generator, false)
  
  # Stack them all together
  img_grid = vcat([hcat(imgs...) for imgs in partition(fake_imgs, 6)]...)
end

cd(@__DIR__)

################################ Loss and Training ##############################
# binary cross entropy
function bce(ŷ, y)
  neg_abs = -abs.(ŷ)
  mean(relu.(ŷ) .- ŷ .* y .+ log.(1 + exp.(neg_abs)))
end

function train(x)
  global training_steps

  z = rand(dist, noise_dim, BATCH_SIZE) |> gpu
  inp = 2x - 1 |> gpu
 
  zero_grad!(discriminator)
 
  D_real = discriminator(inp)
  D_real_loss = bce(D_real, ones(D_real.data))

  fake_x = generator(z)
  D_fake = discriminator(fake_x)
  D_fake_loss = bce(D_fake, zeros(D_fake.data))

  D_loss = D_real_loss + D_fake_loss

  Flux.back!(D_loss)
  opt_disc()

  zero_grad!(generator)
  
  z = rand(dist, noise_dim, BATCH_SIZE) |> gpu
  fake_x = generator(z)
  D_fake = discriminator(fake_x)
  G_loss = bce(D_fake, ones(D_fake.data))

  Flux.back!(G_loss)
  opt_gen()

  if training_steps % verbose_freq == 0
    println("D Loss: $(D_loss.data) | G loss: $(G_loss.data)")
  end

  training_steps += 1
  param(0.0f0)
end

#evalcb = throttle(() -> (save("sample_dcgan.png", sample()); println("Sample saved")), 100)
#@epochs 4 Flux.train!(train, zip(data), SGD(params(discriminator), 0.0f0), cb=evalcb)

for e = 1:NUM_EPOCHS
  for imgs in data
    train(imgs)
  end
  println("Epoch $e over.")
end

save("sample_dcgan.png", sample())
