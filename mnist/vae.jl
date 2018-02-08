using Flux, Flux.Data.MNIST, PyPlot
using Flux: throttle, params
using Juno: @progress

rng = MersenneTwister(1234567)

# Add an iterator to simply run over the data set.
struct MinibatchIterator{TX<:AbstractMatrix}
  X::TX
  M::Int
end
Base.start(d::MinibatchIterator) = 1
function Base.next(d::MinibatchIterator, p::Int)
  p′ = min(p + d.M, size(d.X, 2))
  return (view(d.X, :, p:p′-1),), p′
end
Base.done(d::MinibatchIterator, p::Int) = p == size(d.X, 2)

# Extend distributions slightly to have a numerically stable logpdf for `p` close to 1 or 0.
using Distributions
import Distributions: logpdf
logpdf(b::Bernoulli, y::Bool) = y * log(b.p + eps()) + (1 - y) * log(1 - b.p + eps())

# Load data, binarise it, and partition into mini-batches of M.
X = float(hcat(vec.(MNIST.images())...))
X[X .> 0.5] = 1
X[X .< 1] = 0
X = convert(Matrix{Bool}, X)
N, M = size(X, 2), 100
data = MinibatchIterator(X, M)


################################# Define Model #################################

# Latent dimensionality, # hidden units.
Dz, Dh = 5, 500

# Components of recognition model / "encoder" MLP.
A, μ, logσ = Dense(28^2, Dh, tanh), Dense(Dh, Dz), Dense(Dh, Dz)
g(X) = (h = A(X); (μ(h), logσ(h)))
z(μ, logσ) = μ + exp(logσ) * randn(rng)

# Generative model / "decoder" MLP.
f = Chain(Dense(Dz, Dh, tanh), Dense(Dh, 28^2, σ))


####################### Define ways of doing things with the model. #######################

# KL-divergence between approximation posterior and N(0, 1) prior.
kl_q_p(μ, logσ) = 0.5 * sum(exp.(2 .* logσ) + μ.^2 - 1 .+ logσ.^2)

# logp(x|z) - conditional probability of data given latents.
logp_x_z(x, z) = sum(logpdf.(Bernoulli.(f(z)), x))

# Monte Carlo estimator of mean ELBO using M samples.
L̄(X) = ((μ̂, logσ̂) = g(X); (logp_x_z(X, z.(μ̂, logσ̂)) - kl_q_p(μ̂, logσ̂)) / M)

# Sample from the learned model.
sample(M::Int=1) = rand.(Bernoulli.(f(z.(zeros(Dz, M), zeros(Dz, M)))))


################################# Learn Parameters ##############################

evalcb = throttle(() -> @show(-L̄(X[:, rand(1:60000, M)])), 30);
opt = ADAM(vcat(params(A), params(μ), params(logσ), params(f)));
@progress for i = 1:15
  info("Epoch $i")
  Flux.train!(X->-L̄(X) + 0.5 * sum(x->sum(x.^2), params(f)), data, opt, cb=evalcb)
end
