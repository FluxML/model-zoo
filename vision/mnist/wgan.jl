using Flux, Flux.Data.MNIST
using Flux: @epochs, back!, testmode!, throttle
using Base.Iterators: partition
using NNlib: relu, leakyrelu
using Distributions: Uniform
using CUDAnative:tanh
using CuArrays

# Encode MNIST images as compressed vectors that can later be decoded back into
# images.
BATCH_SIZE = 128
training_step = 0
c = 0.01f0
gen_update_frq = 5  # Updates generators every 5 training steps

imgs = MNIST.images()

# Partition into batches of size 128
data = [float(hcat(vec.(imgs)...)) for imgs in partition(imgs, BATCH_SIZE)]
#data = gpu.(data)

NUM_EPOCHS = 50
noise_dim = 100
channels = 128
hidden_dim = 7 * 7 * channels

dist = Uniform(-1, 1)
############################### WGAN Architecture ##############################
################################## Generator ###################################

fc_gen = Chain(Dense(noise_dim, 1024), BatchNorm(1024, relu),
            Dense(1024, hidden_dim), BatchNorm(hidden_dim, relu))
deconv_ = Chain(ConvTranspose((4,4), channels=>64;stride=(2,2),pad=(1,1)), BatchNorm(64, relu),
                ConvTranspose((4,4), 64=>1, tanh;stride=(2,2), pad=(1,1)))

generator = Chain(fc_gen..., x -> reshape(x, 7, 7, channels, :), deconv_...) |> gpu

################################## Discriminator ###############################

fc_disc = Chain(Dense(hidden_dim, 1024), BatchNorm(1024), 
					 x->leakyrelu.(x, 0.2f0), Dense(1024, 1))
conv_ = Chain(Conv((4,4), 1=>64;stride=(2,2), pad=(1,1)), x->leakyrelu.(x, 0.2f0),
             Conv((4,4), 64=>channels; stride=(2,2), pad=(1,1)), BatchNorm(channels), 
			 x->leakyrelu.(x, 0.2f0))

discriminator = Chain(conv_..., x->reshape(x, hidden_dim, :), fc_disc...) |> gpu

################################################################################

opt_gen  = ADAM(params(generator), 0.001f0; β1 = 0.5f0)
opt_disc = ADAM(params(discriminator), 0.001f0; β1 = 0.5f0)

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

############################ Saving generated images ###########################
using Images

img(x) = Gray.(reshape((x+1)/2, 28, 28))

function sample()
  # 36 random digits
  noise = [rand(dist, noise_dim, 1) |> gpu for i=1:36]
 
  testmode!(generator)
  fake_imgs = img.(map(x -> cpu(generator(x).data), noise))
  testmode!(generator, false)

  vcat([hcat(imgs...) for imgs in partition(fake_imgs, 6)]...)
end

cd(@__DIR__)

################################################################################

function train(x)
  global training_step
  z = rand(dist, noise_dim, BATCH_SIZE) |> gpu
  inp = reshape(2x - 1, 28, 28, 1, :) |> gpu

  zero_grad!(discriminator)
 
  D_real = discriminator(inp)
  D_real_loss = -mean(D_real)

  fake_x = generator(z)
  D_fake = discriminator(fake_x)
  D_fake_loss = mean(D_fake)

  D_loss = D_real_loss + D_fake_loss

  Flux.back!(D_loss)
  opt_disc()

  for p in params(discriminator)
    p.data .= clamp.(p.data, -c, c)
  end
 
  if (training_step+1) % gen_update_frq == 0
    zero_grad!(generator)
    z = rand(dist, noise_dim, BATCH_SIZE) |> gpu
    fake_x = generator(z)
    D_fake = discriminator(fake_x)
    G_loss = -mean(D_fake)
    Flux.back!(G_loss)
    opt_gen()

    println("D loss: $(D_loss.data) | G loss: $(G_loss.data)")
  end

  training_step += 1
  #param(1.0f0)
end

#evalcb = throttle(() -> (save("sample_wgan.png", sample()); println("Sample saved")), 25)
#@epochs 50 Flux.train!(train, zip(data), SGD(params(generator), 0.0f0), cb=evalcb)

for e = 1:NUM_EPOCHS
  for imgs in data
    train(imgs)
  end
  println("Epoch $e over.")
end

save("sample_wgan.png", sample())
