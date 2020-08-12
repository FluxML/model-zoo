# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
using Flux, Flux.Data.MNIST, Statistics, Flux.Optimise
using Flux: throttle, params
using Images


# %%
X = (float.(hcat(vec.(MNIST.images())...)) .> 0.5) 


# %%
N, M = size(X, 2), 100


# %%
data = [X[:,i] for i in Iterators.partition(1:N,M)]


# %%
Dz, Dh = 5, 500
A, μ, logσ = Dense(28^2, Dh, tanh) , Dense(Dh, Dz) , Dense(Dh, Dz) 


# %%
g(X) = (h = A(X); (μ(h), logσ(h)))


# %%
function sample_z(μ, logσ)
    eps = randn(Float32, size(μ)) 
    return μ + exp.(logσ) .* eps
end


# %%
f = Chain(Dense(Dz, Dh, tanh), Dense(Dh, 28^2, σ))


# %%
kl_q_p(μ, logσ) = 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1f0 .+ logσ.^2)


# %%
function logp_x_z(x, z)
    p = f(z)
    ll = x .* log.(p .+ eps(Float32)) + (1f0 .- x) .* log.(1 .- p .+ eps(Float32))
    return sum(ll)
end


# %%
L̄(X) = ((μ̂, logσ̂) = g(X); (logp_x_z(X, sample_z(μ̂, logσ̂)) - kl_q_p(μ̂, logσ̂)) * 1 // M)


# %%
loss(X) = -L̄(X) + 0.01f0 * sum(x->sum(x.^2), params(f))


# %%
function modelsample()  
  ϕ = zeros(Float32, Dz)
  p = f(sample_z(ϕ, ϕ))
  u = rand(Float32, size(p))
  return (u .< p) 
end


# %%
evalcb = throttle(() -> @show(-L̄(X[:, rand(1:N, M)])), 10)
opt = ADAM()
ps = params(A, μ, logσ, f)


# %%
for i = 1:10
  @info "Epoch $i"
  Flux.train!(loss, ps, zip(data), opt, cb=evalcb)
end


# %%
img(x) = Gray.(reshape(x, 28, 28))
sample = hcat(img.([modelsample() for i = 1:10])...)


# %%
sample


# %%
save("sample.png", sample)


# %%



