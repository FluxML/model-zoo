using Flux, Statistics
using Flux.Optimise: update!
include("relax.jl")

D = 100
θ = param(zeros(D))
c = collect(range(0,1,length = D))
f(x) = sum((x .- c).^2, dims = 1)
b = BernoulliRelax(f, Chain(Dense(D, 5, relu), Dense(5, 1)))

ϕ = params(b)
push!(ϕ, θ)
nsamples = 10

opt = ADAM(0.1)
@elapsed for i in 1:2000
    fVal = ∇relax!(θ, b, nsamples)
    update!(opt, ϕ)   
    mod(i, 10) == 0 && println(i, " fVal = ", fVal)
end

