using Flux, Flux.Data.MNIST
using Flux: @epochs, back!
using Base.Iterators: partition
using Juno: @progress
using NNlib: relu, leakyrelu
# using CuArrays

# Encode MNIST images as compressed vectors that can later be decoded back into
# images.
BATCH_SIZE = 1000
training_step = 0
c = 0.01f0
gen_update_frq = 5  # Updates generators every 5 training steps

imgs = MNIST.images()

# Partition into batches of size 1000
data = [float(cat(4, reshape(imgs, size(imgs)..., 1)...)) for imgs in partition(imgs, BATCH_SIZE)]
#data = gpu.(data)


data_dim = size(data[1], 1)
hidden_dim2 = 7 * 7 * 128

############################### WGAN Architecture ##############################
################################## Generator ###################################
fc_gen = Chain(Dense(62, 1024), BatchNorm(1024, NNlib.relu),
            Dense(1024, hidden_dim2), BatchNorm(hidden_dim2, NNlib.relu))
deconv_ = Chain(ConvTranspose((4,4), 128=>64;stride=(2,2),pad=(1,1)), BatchNorm(64, NNlib.relu),
                ConvTranspose((4,4), 64=>1, tanh;stride=(2,2),pad=(1,1)))

generator = Chain(fc_gen..., x->reshape(x, 7, 7, 128, :), deconv_...)

################################## Discriminator ###############################
fc_disc = Chain(Dense(hidden_dim2, 1024), BatchNorm(1024), x->leakyrelu.(x,0.2f0),
            Dense(1024, 1))
conv_ = Chain(Conv((4,4), 1=>64;stride=(2,2),pad=(1,1)), x->leakyrelu.(x,0.2f0),
             Conv((4,4), 64=>128;stride=(2,2),pad=(1,1)),
             BatchNorm(128), x->leakyrelu.(x,0.2f0))

discriminator = Chain(conv_..., x->reshape(x, hidden_dim2, :), fc_disc...)
################################################################################

opt_gen = ADAM(params(generator))
opt_disc = ADAM(params(discriminator))

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
################################################################################

function train(x)
  global training_step
  z = rand(62, BATCH_SIZE)
  zero_grad!(discriminator)

  D_real = discriminator(x)
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

  if training_step % gen_update_frq == 0
    zero_grad!(generator)
    fake_x = generator(z)
    D_fake = discriminator(fake_x)
    G_loss = -mean(D_fake)
    Flux.back!(G_loss)
    opt_gen()
  end
  training_step += 1
  @show training_step, D_loss
end

@epochs 1 train(data[1])
# Sample output

using Images

img(x) = Gray.(clamp.(x, 0, 1))

function sample()
  # 20 random digits
  before = [rand(62, 1) for i=1:10]
  # Before and after images
  after = img.(map(x -> generator(x).data, before))
  # Stack them all together
  hcat(after...)
end

cd(@__DIR__)

save("sample.png", sample())
