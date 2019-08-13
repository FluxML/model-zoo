mutable struct PRelu{T}
	α::T
end

@treelike PRelu

PRelu(num_channels::Int;init=Flux.glorot_uniform) = PRelu(param(init(num_channels)))

function (m::PRelu)(x)
	size(x)[end - 1] == length(m.α) || error("NUmber of channels in input does not match with length of α")
	max.(0.0f0,x) .+ (reshape(m.α,ones(Int64,length(size(x)) - 2)...,length(m.α),1) .* min.(0.0f0,x))
end

function split_channels(x::AbstractArray,val::Int) # Split chaannels into partitions, each containing `val` elements
    indices = collect(1:size(x)[end-1])
    channels_par = partition(indices,div(size(x)[end-1],val))

    out = []
    for c in channels_par
       c = [c_ for c_ in c] # Extremely hacky, temporary line of code
       push!(out,x[:,:,c,:])
    end
    return out
end

function phase_shift(x,r)
    W,H,C,N = size(x)
    x = reshape(x,W,H,r,r,N)
    x = [x[i,:,:,:,:] for i in 1:W]
    x = cat([t for t in x]...,dims=2)
    x = [x[i,:,:,:] for i in 1:size(x)[1]]
    x = cat([t for t in x]...,dims=2)
    x
end

function PixelShuffle(x,r=3)
	length(size(x)) == 4 || error("Size of array for PixelShuffle must be 4")
	size(x)[end-1] % (r*r) == 0 || error("Number of channels must be divisible by r*r")

    C_out = div(size(x)[end-1],r*r)
    sch = split_channels(x,C_out)
    out = cat([phase_shift(c,r) for c in split_channels(x,C_out)]...,dims=3)
    reshape(out,size(out)[1],size(out)[2],C_out,div(size(out)[end],C_out))
end
