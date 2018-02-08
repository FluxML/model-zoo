using Flux, Flux.Data.MNIST, PyPlot
using Flux: throttle, params
using Juno: @progress

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

# Latent dimensionality, # hidden units, and minibatch sizes resp.
Dz, Dh = 5, 500
rng = MersenneTwister(1234567)

# Recognition model / "decoder" MLP.
g = Dense(28^2, Dh, tanh)
μ, logσ = Dense(Dh, Dz), Dense(Dh, Dz)
z(μ, logσ) = μ + exp(logσ) * randn(rng)

# Generative model / "encoder" MLP.
f = Chain(Dense(Dz, Dh, tanh), Dense(Dh, 28^2, σ))

# Compute ELBO.
kl(μ, logσ) = 0.5 * sum(exp.(2 .* logσ) + μ.^2 - 1 .+ logσ.^2)
L̂(X, Xpr) = sum(logpdf.(Bernoulli.(Xpr), X))
function nveELBO(X)
  h = g(X)
  μ̂, logσ̂ = μ(h), logσ(h)
  ẑ = z.(μ̂, logσ̂)
  return -(L̂(X, f(ẑ)) - kl(μ̂, logσ̂)) / M
end


################################# Learn Parameters ##############################

evalcb = throttle(() -> @show(nveELBO(X[:, rand(1:60000, M)])), 5)
opt = ADAM(vcat(params(g), params(μ), params(logσ), params(f)))
@progress for i = 1:10
  info("Epoch $i")
  Flux.train!(nveELBO, data, opt, cb=evalcb)
end


################################# Sample Images #################################

# Sample from the learned model.
sample(M::Int=1) = rand.(Bernoulli.(f(z.(zeros(Dz, M), zeros(Dz, M)))))

# Get the parameters of the approximate posterior.
encode(X) = (h = g(X); (μ(h), logσ(h)))

