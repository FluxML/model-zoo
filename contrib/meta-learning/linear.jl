function xavier_uniform(dims...) 
    bound = sqrt(1 / dims[2])
    return Float32.(rand(Uniform(-bound, bound), dims...))
end

struct Linear{F,S,T}
    W::S
    b::T
    σ::F
end

Linear(W, b) = Linear(W, b, identity)

function Linear(in::Integer, out::Integer, σ = identity;
                initW = xavier_uniform, initb = nothing)
    if initb == nothing
        bias_bound = 1 / sqrt(in)
        initb = (out) -> Float32.(rand(Uniform(-bias_bound, bias_bound), out))
    end
    return Linear(param(initW(out, in)), param(initb(out)), σ)
end

Flux.@treelike Linear

function (a::Linear)(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    σ.(W*x .+ b)
end

function Base.show(io::IO, l::Linear)
    print(io, "Linear(", size(l.W, 2), ", ", size(l.W, 1))
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end
