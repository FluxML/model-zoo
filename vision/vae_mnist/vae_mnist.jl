# Variational Autoencoder(VAE)
#
# Auto-Encoding Variational Bayes
# Diederik P Kingma, Max Welling
# https://arxiv.org/abs/1312.6114


#===== PACKAGES =====#

using Flux, MLDatasets, ProgressMeter, Random
using Images: Gray, save
using JLD2  # recommended way to save model state
using CUDA  # to use a GPU, although this model does not require one


#===== MODEL =====#

# The model has an encoder and a decoder. Our encoder doesn't fit any of the built-in Flux layers,
# so we define a new layer. At this point we don't commit to specific sizes `input_dim, latent_dim, hidden_dim`.

# 1. Define a structure to contain the pieces:
struct Encoder
    linear
    mean
    logstd
end
# (Adding types `struct Encoder{A,B,C}; linear::A; ...` is generally good practice,
# but here produces no detectable speedup.)

# 2. Tell Flux to look for parameters inside this struct:
Flux.@layer :expand Encoder

# 3. Make the struct callable to define the forward pass:
function (encoder::Encoder)(x)
    h = encoder.linear(x)
    encoder.mean(h), encoder.logstd(h)  # returns a Tuple
end

# 4. Write a constructor method which initialises given the sizes:
Encoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Encoder(
    Dense(input_dim => hidden_dim, tanh),   # linear
    Dense(hidden_dim => latent_dim),        # μ
    Dense(hidden_dim => latent_dim),        # logσ
)

# The decoder is just a multi-layer perceptron, here's a function which constructs it
# using the same sizes as Encoder:

make_decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Chain(
    Dense(latent_dim => hidden_dim, tanh),
    Dense(hidden_dim => input_dim),
)

# Here is how the encoded and decoder are used: The encoder produces mean & std,
# which are used to generate random y, which is the input to the decoder.
# Type restrictions `enc::Encoder, dec::Chain` are basically documentation here,
# and ensure we can't mix up the order of arguments.

function reconstuct(enc::Encoder, dec::Chain, x::AbstractArray)
    μ, logσ = enc(x)
    y = μ + exp.(logσ) .* randn.(Float32)  # randn.() is a new random number at every entry
    z = dec(y)
    (; μ, logσ, z)  # returns a NamedTuple
end

function model_loss(enc::Encoder, dec::Chain, x::AbstractArray)
    μ, logσ, z = reconstuct(enc, dec, x)
    len = size(x)[end]

    # KL-divergence
    kl_q_p = sum(@.(exp(2*logσ) + μ^2 - 1 - 2*logσ)) / (2*len)

    logp_x_z = -Flux.logitbinarycrossentropy(z, x, agg=sum) / len

    # # L2 regularization
    # reg = l2_reg/2 * sum(x->sum(x.^2), Flux.params(decoder))
    
    -logp_x_z + kl_q_p # + reg
end

# We're going to add L2 regularisation, to this loss, but will do this efficiently
# by using WeightDecay during training, rather than adding an explicit penalty to the loss.


#===== DATA =====#

# Calling MLDatasets.MNIST() will dowload the dataset if necessary,
# and return a struct containing it.
# It takes a few seconds to read from disk each time, so do this once:

train_data = MLDatasets.MNIST()  # i.e. split=:train
test_data = MLDatasets.MNIST(split=:test)

# train_data.features is a 28×28×60000 Array{Float32, 3} of the images.
# train_data.targets has the labels, but we won't use those here.

# We need a 2D array for our model, each column of which is one image. Let's reshape now:

train_matrix = reshape(train_data.features, 28^2, :);

# We're going to save PNG files showing many images as small tiles. Here's a function
# which assembles them into a given number of rows:

function image_tiles(x::AbstractMatrix, n_rows::Int)
    rows = Flux.chunk(x |> cpu, n_rows)
    tmp = reshape.(rows, 28, :)
    Gray.(permutedims(vcat(tmp...)))
end

image_tiles(train_matrix[:, 1:6], 2);  # 84×56 Array{Gray{Float32},2}


#===== TRAINING =====#

# Let's collect all the ...

args = (;
    # about training:
    learn_rate = 0.001,      # learning rate
    l2_reg = 0.02,           # regularization paramater
    epochs = 20,
    batch_size = 128,

    # about the model:
    device = gpu,            # this is ignored if you don't have a GPU
    seed = 0,                # set to >0 to always use the same random seed
    input_dim = 28^2,        # image size, standard MNIST
    latent_dim = 2,          # latent dimension of encoding
    hidden_dim = 500,        # hidden dimension within both encoder and decoder

    # saved output:
    sample_sqrt = 10,        # make sample_sqrt^2 outputs at a time    
    save_path = "output",    # images will be saved as e.g. "output/original.png"
)


# Initialise the model, and move to GPU if using one:

if args.seed > 0
    Random.seed!(args.seed)
end
encoder = Encoder(args.input_dim, args.latent_dim, args.hidden_dim) |> args.device
decoder = make_decoder(args.input_dim, args.latent_dim, args.hidden_dim) |> args.device

# Initialise the optimiser state -- Adam remembers some momenta between steps.
# We add L2 regularisation only to the decoder...

opt_enc = Flux.setup(Adam(args.learn_rate), encoder)

rule_dec = OptimiserChain(WeightDecay(args.l2_reg), Adam(args.learn_rate))
opt_dec = Flux.setup(rule_dec, decoder)

# Load training data.
# Note that `DataLoader(...) |> gpu` will move one batch at a time to the GPU,
# and that `shuffle=true` re-shuffles each epoch.

loader = Flux.DataLoader(train_matrix; batchsize=args.batch_size, shuffle=true) |> args.device


# fixed input
original = train_matrix[:, 1:args.sample_sqrt^2] |> args.device;
image_path = joinpath(args.save_path, "original.png")
save(image_path, image_tiles(original, args.sample_sqrt))


# Main training loop

for epoch = 1:args.epochs
    println("Starting epoch $epoch of ", args.epochs)
    progress = ProgressMeter.Progress(length(loader))

    for x in loader
        loss, (grad_enc, grad_dec, _) = Flux.withgradient(model_loss, encoder, decoder, x)
        Flux.update!(opt_enc, encoder, grad_enc)
        Flux.update!(opt_dec, decoder, grad_dec)

        # Fancy progress meter, printing the loss as we go along:
        ProgressMeter.next!(progress; showvalues=[(:loss, round(loss, sigdigits=3))]) 
    end

    # Once per epoch, we save the image
    _, _, rec_original = reconstuct(encoder, decoder, original)
    rec_original = sigmoid.(rec_original)

    image = image_tiles(rec_original, args.sample_sqrt)
    image_path = joinpath(args.save_path, "epoch_$(epoch).png")
    save(image_path, image)
    println("Image saved: ", image_path)
end


# Save the fully trained model's state, after moving it off the GPU:

encoder_state = Flux.state(encoder) |> cpu
decoder_state = Flux.state(decoder) |> cpu
model_path = joinpath(args.save_path, "model.jld2")
JLD2.jldsave(model_path; encoder_state, decoder_state, args)
@info "Model saved!" model_path


#===== LOADING =====#

# First we load args, for e.g. the hidden dimensions used, then we re-build the model:
args2 = JLD2.load("output/model.jld2", "args")
enc2 = Encoder(args2.input_dim, args2.latent_dim, args2.hidden_dim)
dec2 = make_decoder(args2.input_dim, args2.latent_dim, args2.hidden_dim)

# Check that the new model runs but produces junk:
_, _, rec_original = reconstuct(enc2, dec2, original |> cpu)
image = image_tiles(sigmoid.(rec_original), args.sample_sqrt)
save(joinpath(args.save_path, "untrained.png"), image)

# Now load the model state into this new model:
enc_st_2 = JLD2.load("output/model.jld2", "encoder_state")
Flux.loadmodel!(enc2, enc_st_2)
dec_st_2 = JLD2.load("output/model.jld2", "decoder_state")
Flux.loadmodel!(dec2, dec_st_2)

# Check that the model now works:
_, _, rec_original = reconstuct(enc2, dec2, original |> cpu)
image = image_tiles(sigmoid.(rec_original), args.sample_sqrt)
save(joinpath(args.save_path, "reloaded.png"), image)


#===== THE END =====#
