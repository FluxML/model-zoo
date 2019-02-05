#   Implementation of relax method for Bernoulli random variable from  the paper 
#   Backpropagation through the Void: Optimizing control variates for black-box gradient estimation,
#   Will Grathwohl, Dami Choi, Yuhuai Wu, Geoffrey Roeder, David Duvenaud 
# 
#   https://arxiv.org/abs/1711.00123
#
#   The implementation is a copy-cat from https://github.com/duvenaud/relax

using Flux, Statistics
using Flux.Tracker

expit(x) = 1 / (1 + exp(-x))

logit(x) = log(x / (1 - x))

heaviside(z) = z >= 0

bernoulli_softmax(z, log_temperature) = expit(z / exp(log_temperature))

logistic_sample(noise, mu = 0, sigma=1) = mu + logit(noise) * sigma

function logaddexp(x, y) 
    m =  max(x, y)
    log(exp(x - m) + exp(y - m)) + m
end

function logistic_logpdf(x, mu = 0, scale = 1)
    y = (x - mu) / (2 * scale)
    -2 * logaddexp(y, -y) - log(scale)
end

bernoulli_sample(logit_theta, noise) = logit(noise) < logit_theta

# relaxed_bernoulli_sample(logit_theta, noise, log_temperature) = bernoulli_softmax(logistic_sample(noise, logit_theta), log_temperature)
relaxed_bernoulli_sample(logit_theta, noise, log_temperature) = bernoulli_softmax(logistic_sample(noise, expit.(logit_theta)), log_temperature)

"""
    Computes p(u|b), where b = H(z), z = logit_theta + logit(noise), p(u) = U(0, 1)
"""
function conditional_noise(logit_theta, samples, noise)
    uprime = expit.(-logit_theta)
    @. samples * (noise * (1 - uprime) + uprime) + (1 - samples) * noise * uprime
end

"""
    log Bernoulli(targets | theta), targets are 0 or 1.
"""
bernoulli_logprob(logit_theta, targets) = -logaddexp(0, -logit_theta * (targets * 2 - 1))


"""
    function reinforce(θ, u, x)

    θ, u, x =  param, noise, func_vals
"""
function reinforce(θ, noise, x)
    samples = bernoulli_sample.(θ, noise)
    x .* Flux.gradient((θ) -> sum(bernoulli_logprob.(θ, samples)), θ; nest = true)[1]
end


function concrete(θ, log_temperature, noise, f)
    relaxed_samples = relaxed_bernoulli_sample.(θ, noise, log_temperature)
    f(relaxed_samples)
end


function relax(θ, u, v, log_temperature, surrogate, f)
    samples = bernoulli_sample.(θ, u)
    ∇surrogate = Flux.gradient(θ -> sum(concrete(θ, log_temperature, u, surrogate)), θ; nest = true)[1]

    cond_noise = conditional_noise.(θ, samples, v)  # z tilde
    cond_surrogate = concrete(θ, log_temperature, cond_noise, surrogate)
    ∇cond_surrogate = Flux.gradient(() -> sum(cond_surrogate), Params(θ); nest = true)[θ]

    func_vals = f(samples)
    (func_vals, reinforce(θ, u, func_vals .- cond_surrogate) .+ ∇surrogate .- ∇cond_surrogate)
end

"""

    est_params = (τ, ϕ)
    θ --- parameters of Bernoulli probability distribution
    ϕ --- parameters of surrogate network
"""
function relax_all(θ, ϕ, τ, surrogate, u, v, f)
    # Returns objective, gradients, and gradients of variance of gradients.
    θm = param(repeat(Flux.data(θ), inner = (1, size(u, 2))))
    func_vals, ∂f_∂θ = relax(θm, u, v, τ, surrogate, f)
    ∂var_∂ϕ = Flux.gradient(() -> sum(∂f_∂θ .^ 2) ./ size(u, 2), Params(ϕ); nest = true)
    
    for p in ϕ
        p.grad .= Flux.data(∂var_∂ϕ[p])
    end
    θ.grad .= dropdims(mean(Flux.data(∂f_∂θ), dims = 2), dims = 2)
    func_vals
end

struct BernoulliRelax{F, S, T}
    f::F
    surrogate::S
    τ::T
    ϕ::Params
end

Flux.@treelike(BernoulliRelax)

function BernoulliRelax(f, surrogate)
    τ = param([0.0])
    ϕ = params(surrogate)
    push!(ϕ, τ)
    BernoulliRelax(f, surrogate, τ, ϕ)
end

function (m::BernoulliRelax)(θ, u, v)
    θm = param(repeat(Flux.data(θ), inner = (1, size(u, 2))))
    func_vals, ∂f_∂θ = relax(θm, u, v, m.τ, surrogate, m.f)
    ∂var_∂ϕ = Flux.gradient(() -> sum(∂f_∂θ .^ 2) ./ size(u, 2), Params(ϕ); nest = true)
    
    foreach(p -> p.grad .= Flux.data(∂var_∂ϕ[p]), ϕ)
    θ.grad .= dropdims(mean(Flux.data(∂f_∂θ), dims = 2), dims = 2)
    func_vals
end


D = 100
θ = param(zeros(D))
c = collect(range(0,1,length = D))
f(x) = sum((x .- c).^2, dims = 1)

b = BernoulliRelax(f, Chain(Dense(D, 5, relu), Dense(5, 1)))
ϕ = params(b)
nsamples = 10

opt = ADAM(0.1)
for i in 1:2000
    fVal = b(θ, rand(D, nsamples), rand(D, nsamples))
    Flux.Optimise.update!(opt, ϕ)
    Flux.Optimise.update!(opt, [θ])
    mod(i, 10) == 0 && println(i, " fVal = ", mean(fVal))
end