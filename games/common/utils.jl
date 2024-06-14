# weight initialization
function _random_normal(shape...)
    return map(Float32,rand(Normal(0,0.1),shape...))
end

function constant_init(shape...)
    return map(Float32,ones(shape...) * 0.1)
end

function normalise(arr)
    (arr .- mean(arr))./(sqrt(var(arr) + 1e-10))
end

function normalise_across_procs(arr)
    arr = reshape(arr,EPISODE_LENGTH,num_processes)
    reshape((arr .- mean(arr,dims=2))./(sqrt.(var(arr,dims=2) .+ 1e-10)),1,EPISODE_LENGTH*num_processes)
end

"""
Returns a Generalized Advantage Estimate for an episode
"""

function gae(policy,states::Array,actions::Array,rewards::Array,next_states::Array,num_steps::Int;γ=0.99,λ=0.95)
    Â = []
    A = 0.0
    for i in reverse(1:length(states))
        if length(states) < num_steps && i == length(states)
            δ = rewards[i] - policy.value_net(states[i]).data[1]
        else
            δ = rewards[i] + γ*policy.value_net(next_states[i]).data[1] - policy.value_net(states[i]).data[1]
        end

        A = δ + (γ*λ*A)
        push!(Â,A)
    end
    
    Â = reverse(Â)
    return Â
end

"""
Returns the cumulative discounted returns for each timestep
"""

function disconunted_returns(rewards::Array,last_val=0;γ=0.99)
    r = 0.0
    returns = []

    for i in reverse(1:length(rewards))
        r = rewards[i] + γ*r
	if i == length(rewards)
		r = r + last_val
	end
        push!(returns,r)
    end
    returns = reverse(returns)

    returns
end

"""
Flatten gradients and model parameters
"""

function get_flat_grads(gradients,models)
    """
    Flattens out the gradients and concatenates them

    models : An array of models whose parameter gradients are to be falttened
    
    Returns : Tracker Array of shape (NUM_PARAMS,1)
    """

    flat_grads = []

    function flatten!(p)
        if typeof(p) <: TrackedArray
            prod_size = prod(size(p))
            push!(flat_grads,reshape(gradients[p],prod_size))
        end
    end
    
    for model in models
        mapleaves(flatten!,model)
    end
    
    flat_grads = cat(flat_grads...,dims=1)
    flat_grads = reshape(flat_grads,length(flat_grads),1)
    
    return flat_grads
end

function get_flat_params(models)
    """
    Flattens out the parameters and concatenates them

    models : An array of models whose parameters are to be falttened
    
    Returns : Tracker Array of shape (NUM_PARAMS,1)
    """

    flat_params = []
    
    function flatten!(p)
        if typeof(p) <: TrackedArray
            prod_size = prod(size(p))
            push!(flat_params,reshape(p,prod_size))
        end
    end
    
    for model in models
        mapleaves(flatten!,model)
    end
    
    flat_params = cat(flat_params...,dims=1)
    flat_params = reshape(flat_params,length(flat_params),1)
    
    return flat_params
end

function set_flat_params(parameters,models)
    """
    Sets values of `parameters` to the `model`
    
    parameters : flattened out array of model parameters
    models : an array of models whose parameters are to be set
    """
    ptr = 1
    
    function assign!(p)
        if typeof(p) <: TrackedArray
            prod_size = prod(size(p))
            
            p.data .= Float32.(reshape(parameters[ptr : ptr + prod_size - 1,:],size(p)...)).data
            ptr += prod_size
        end
    end
    
    for model in models
        mapleaves(assign!,model)
    end
    
    print("")
end
