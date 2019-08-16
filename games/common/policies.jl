using BSON:@save,@load

struct EnvWrap{T,V,W}
    env::T
    STATE_SIZE::V
    ACTION_SIZE::V
    ENV_NAME::W
end

function EnvWrap(env_name::String)
    env = make(env_name,:rgb)
    
    if typeof(env._env.observation_space) <: Gym.Space.Discrete
        STATE_SIZE = env._env.observation_space.n
    elseif typeof(env._env.observation_space) <: Gym.Space.Box
        STATE_SIZE = env._env.observation_space.shape[1]
    else
        error("Typeof environment is not supported")
    end

    if typeof(env._env.action_space) <: Gym.Space.Discrete
        ACTION_SIZE = env._env.action_space.n
    elseif typeof(env._env.action_space) <: Gym.Space.Box
        ACTION_SIZE = env._env.action_space.shape[1]
    else
        error("Typeof environment is not supported")
    end

    return EnvWrap(env,STATE_SIZE,ACTION_SIZE,ENV_NAME)
end

function scale_rewards(env_wrap::EnvWrap,rewards)
    if env_wrap.ENV_NAME == "Pendulum-v0"
         rewards = rewards ./ 16.2736044 .+ 2.0f0
	 # rewards = rewards ./ 16.2736044
    end
    
    rewards
end

"""
Define all policies here
"""

"""
---------------Categorical Policy----------------
---------- For discrete action spaces -----------
"""

mutable struct CategoricalPolicy
    π # Neural network for the policy
    value_net # Neural network for the value function
    env_wrap # A wrapper for environment variables
end

function CategoricalPolicy(env_wrap::EnvWrap, policy_net = nothing, value_net = nothing)
    if policy_net == nothing
        policy_net = Chain(Dense(
                        env_wrap.STATE_SIZE,30,relu;initW = _random_normal,initb=constant_init),
                        Dense(30,env_wrap.ACTION_SIZE;initW = _random_normal,initb=constant_init),
                        x -> softmax(x))
    end
    
    if value_net == nothing
        value_net = Chain(Dense(env_wrap.STATE_SIZE,30,relu;initW=_random_normal),
                  Dense(30,30,relu;initW=_random_normal),
                  Dense(30,1;initW=_random_normal))
    end
    
    return CategoricalPolicy(policy_net,value_net,env_wrap)
end

"""
-----------Diagonal Gaussian Policy----------
-------- For continuous action space --------
"""

mutable struct DiagonalGaussianPolicy
    μ # Neural network for the mean of the Gaussian distribution
    logΣ # Neural network for the log standard deviation of the Gaussian distribution
    value_net # Neural network for the value function
    env_wrap # A wrapper for environment variables
end

function DiagonalGaussianPolicy(env_wrap::EnvWrap, μ = nothing, value_net = nothing, STD = 0.0)
    if μ == nothing
        μ = Chain(Dense(env_wrap.STATE_SIZE,30,tanh;initW = _random_normal,initb=constant_init),
                     Dense(30,env_wrap.ACTION_SIZE;initW = _random_normal,initb=constant_init),
                     x->tanh.(x),
                     x->param(2.0) .* x)
    end
    
    if value_net == nothing
        value_net = Chain(Dense(env_wrap.STATE_SIZE,30,tanh;initW=_random_normal),
                  Dense(30,30,tanh;initW=_random_normal),
                  Dense(30,1;initW=_random_normal))
    end
    
    logΣ = param(ones(env_wrap.ACTION_SIZE) * STD)

    return DiagonalGaussianPolicy(μ,logΣ,value_net,env_wrap)
end

"""
Define the following for each policy : 
    `action` : a function taking in the policy variable and giving a particular action according to the environemt
    `log_prob` : a function giving the log probability of an action under the current policy parameters
    `entropy` : a function defining the entropy of the policy distribution

Populate each function with it's appropriate distribution
"""

function action(policy,state)
    """
    policy : A policy type defined in `policy.jl`
    state : output of reset!(env) or step!(env,action)
    """
    a = nothing
    
    if typeof(policy) <: CategoricalPolicy
        action_probs = policy.π(state).data
        action_probs = reshape(action_probs,policy.env_wrap.ACTION_SIZE)
        a = Distributions.sample(1:policy.env_wrap.ACTION_SIZE,Distributions.Weights(action_probs))
    elseif typeof(policy) <: DiagonalGaussianPolicy
        # Our policy outputs the parameters of a Normal distribution
        μ = policy.μ(state)
        μ = reshape(μ,policy.env_wrap.ACTION_SIZE)
        log_std = policy.logΣ
        
        σ² = (exp.(log_std)).^2
        Σ = diagm(0=>σ².data)
        
        dis = MvNormal(μ.data,Σ)
        
        a = rand(dis,policy.env_wrap.ACTION_SIZE)
    else
        error("Policy type not yet implemented")
    end
    
    a
end

function test_action(policy,state)
    """
    policy : A policy type defined in `policy.jl`
    state : output of reset!(env) or step!(env,action)
    """
    a = nothing
    
    if typeof(policy) <: CategoricalPolicy
        action_probs = policy.π(state).data
        action_probs = reshape(action_probs,policy.env_wrap.ACTION_SIZE)
        a = Distributions.sample(1:policy.env_wrap.ACTION_SIZE,Distributions.Weights(action_probs))
    elseif typeof(policy) <: DiagonalGaussianPolicy
        # Use only the mean for prediction
        a = policy.μ(state).data
    else
        error("Policy type not yet implemented")
    end
    
    a
end

function log_prob(policy,states::Array,actions::Array)
    log_probs = nothing

    if typeof(policy) <: CategoricalPolicy
        action_probs = policy.π(states)

        actions_one_hot = zeros(policy.env_wrap.ACTION_SIZE,size(action_probs)[end])
        for i in 1:size(action_probs)[end]
            actions_one_hot[actions[:,i][1],i] = 1.0                
        end

        log_probs = log.(sum((action_probs .+ 1f-5) .* actions_one_hot,dims=1))

    elseif typeof(policy) <: DiagonalGaussianPolicy
        μ = policy.μ(states)
        σ = exp.(policy.logΣ)
        σ² = σ.^2
        log_probs = .-(((actions .- μ).^2)./(2.0 .* σ²)) .- 0.5*log.(sqrt(2 * π)) .- log.(σ)

    else
        error("Not implemented")
    end

    log_probs
end

function entropy(policy,states::Array)
    if typeof(policy) <: CategoricalPolicy
        action_probs = policy.π(states)
        return sum(action_probs .* log.(action_probs .+ 1f-10),dims=1)
    elseif typeof(policy) <: DiagonalGaussianPolicy
        return 0.5 + 0.5 * log(2 * π) .+ policy.logΣ
    else
        error("Not Implemented")
    end
end

function kl_divergence(policy,kl_params,states::Array)
    """
    kl_params:
        - old_log_probs : CategoricalPolicy
        - Array([μ,logΣ]) : DiagonalGaussianPolicy
    """

    if typeof(policy) <: CategoricalPolicy
        old_log_probs = hcat(cat(kl_params...,dims=1)...)

        action_probs = policy.π(states)
        log_probs = log.(action_probs)

        log_ratio = log_probs .- old_log_probs
        kl_div = (exp.(old_log_probs)) .* log_ratio
         
        return -1.0f0 .* sum(kl_div,dims=1)
    
    elseif typeof(policy) <: DiagonalGaussianPolicy
        μ0 = policy.μ(states)
        logΣ0 = policy.logΣ
        μ1 = mus = hcat([kl_params[i][1] for i in 1:length(kl_params)]...)
        logΣ1 = hcat([kl_params[i][2] for i in 1:length(kl_params)]...)

        var0 = exp.(2 .* logΣ0)
        var1 = exp.(2 .* logΣ1)
        pre_sum = 0.5 .* (((μ0 .- μ1).^2 .+ var0) ./ (var1 .+ 1e-8) .- 1.0f0) .+ logΣ1 .- logΣ0
        kl = sum(pre_sum,dims=1)
        return kl
    else
        error("Not implemented")
    end
end

function get_policy_params(policy)
    if typeof(policy) <: CategoricalPolicy
        return params(policy.π)
    elseif typeof(policy) <: DiagonalGaussianPolicy
        return params(params(policy.μ)...,params(policy.logΣ)...)
    end
end

function get_policy_net(policy)
    """
    Returns the policy neural network
    """
    if typeof(policy) <: CategoricalPolicy
        return [policy.π]
    elseif typeof(policy) <: DiagonalGaussianPolicy
        return [policy.μ,policy.logΣ]
    end
end

function get_value_params(policy)
    if typeof(policy) <: CategoricalPolicy
        return params(policy.value_net)
    elseif typeof(policy) <: DiagonalGaussianPolicy
        return params(policy.value_net)
    end
end

function get_value_net(policy)
    """
    Returns the value neural network
    """
    if typeof(policy) <: CategoricalPolicy
        return [policy.value_net]
    elseif typeof(policy) <: DiagonalGaussianPolicy
        return [policy.value_net]
    end
end

function save_policy(policy,path = nothing)
    if path == nothing
        if isdir("../../weights") == false
            mkpath("../../weights")
        end

        if typeof(policy) <: CategoricalPolicy
            π = policy.π
            @save "../../weights/policy_cat.bson" π
        elseif typeof(policy) <: DiagonalGaussianPolicy
            μ = policy.μ
            logΣ = policy.logΣ
            @save "../../weights/policy_mu.bson" μ
            @save "../../weights/policy_sigma.bson" logΣ
        end

        value_net = policy.value_net
        @save "../../weights/value.bson" value_net
    else
        @assert typeof(path) <: String "Path must be a string"

        if isdir(path) == false
            mkpath(path)
        end

        if typeof(policy) <: CategoricalPolicy
            π = policy.π
            @save string(path,"policy_cat.bson") π
        elseif typeof(policy) <: DiagonalGaussianPolicy
	    println("IN")
	    println(path)
            μ = policy.μ
            logΣ = policy.logΣ
            @save string(path,"policy_mu.bson") μ
            @save string(path,"policy_sigma.bson") logΣ
        end

        value_net = policy.value_net
        @save string(path,"value.bson") value_net
    end

    println("Saved...")
end

function load_policy(env_wrap::EnvWrap,path = nothing)
    if typeof(env_wrap.env._env.action_space) <: Gym.Space.Discrete
        policy = CategoricalPolicy(env_wrap)

        if path == nothing
            @load "../weights/policy_cat.bson" π
	    @load "../weights/value.bson" value_net
        else
            @load string(path,"policy_cat.bson") π
	    @load string(path,"value.bson") value_net
        end
	
        policy.π = π
	policy.value_net = value_net

        println("Loaded")
        return policy

    elseif typeof(env_wrap.env._env.action_space) <: Gym.Space.Box
        policy = DiagonalGaussianPolicy(env_wrap)

        if path == nothing
            @load "../weights/policy_mu.bson" μ
            @load "../weights/policy_sigma.bson" logΣ
	    @load "../weights/value.bson" value_net
        else
            @load string(path,"policy_mu.bson") μ
            @load string(path,"policy_sigma.bson") logΣ
	    @load string(path,"value.bson") value_net
        end

        policy.μ = μ
        policy.logΣ = logΣ
	policy.value_net = value_net

        println("Loaded")
        return policy

    else
        error("Environment type not supported")
    end
end

