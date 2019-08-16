"""
Abstractions to data values
"""

mutable struct Buffer{T}
    exp_dict::T # T : A dictionary
end

Buffer() = Buffer(Dict())

function register(b::Buffer,name::String)
     b.exp_dict[name] = []
end

"""
Add a variable for it's history to be logged
"""

function add(b::Buffer,name::String,value::Any)
    push!(b.exp_dict[name],value)
end

function get!(b::Buffer,name::String)
	return b.exp_dict[name]
end

function clear(b::Buffer,name=nothing)
	if name == nothing
		for key in keys(b.exp_dict)
			b.exp_dict[key] = []
		end
	else
		@assert typeof(name) <: String "Name of key must be a string"
		b.exp_dict[name] = []
	end

	return nothing
end
