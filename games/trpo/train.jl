using Pkg
Pkg.activate(".")

using Flux
using Gym
import Reinforce.action
import Reinforce:run_episode
import Flux.params
using Flux.Tracker: grad, update!
using Flux: onehot
using Statistics
using Distributed
using Distributions
using LinearAlgebra
using Base.Iterators
using Random
using BSON
using BSON:@save,@load
using JLD

num_processes = 1
include("../common/policies.jl")
include("../common/utils.jl")
include("../common/buffer.jl")
include("rollout.jl")
include("trpo_utils.jl")

function initialize_episode_buffer()
    eb = Buffer()
    register(eb,"states")
    register(eb,"actions")
    register(eb,"rewards")
    register(eb,"next_states")
    register(eb,"dones")
    register(eb,"returns")
    register(eb,"advantages")
    register(eb,"log_probs")
    register(eb,"kl_params")
    
    return eb
end

function initialize_stats_buffer()
    sb = Buffer()
    register(sb,"rollout_returns")
    
    return sb
end

function get_policy(env_wrap::EnvWrap)
    if typeof(env_wrap.env._env.action_space) <: Gym.Space.Discrete
        return CategoricalPolicy(env_wrap)
    elseif typeof(env_wrap.env._env.action_space) <: Gym.Space.Box
        return DiagonalGaussianPolicy(env_wrap)
    else
        error("Policy type not supported")
    end
end

# ----------------Hyperparameters----------------#
# Environment Variables #
# NOTE : TRPO will not work with Categorical Policies as nested AD is not currently defined for `softmax`
ENV_NAME = "Pendulum-v0"
EPISODE_LENGTH = 1000
terminate_horizon = false
resume = false
# Policy parameters #
η = 1e-3 # Learning rate
STD = 0.0 # Standard deviation
# GAE parameters
γ = 0.99
λ = 0.95
# Optimization parameters
δ = 0.01 # KL-Divergence constraint
V_ITERS = 5 # Number of iterations to train the value function network
NUM_EPISODES = 100000
BATCH_SIZE = 256
# FREQUENCIES
SAVE_FREQUENCY = 25
VERBOSE_FREQUENCY = 5
global_step = 0

# Define policy
env_wrap = EnvWrap(ENV_NAME)

if resume == true
    policy = load_policy(env_wrap,"../weights/")
else
    policy = get_policy(env_wrap)
end

# Define buffers
episode_buffer = initialize_episode_buffer()
stats_buffer = initialize_stats_buffer()

# Define optimizer for optimizing value function neural network
opt_value = ADAM(η)

function kl_loss(policy,states::Array,kl_vars)
    return mean(kl_divergence(policy,kl_vars,states))
end

function policy_loss(policy,states::Array,actions::Array,advantages::Array,old_log_probs::Array)
    # Surrogate loss computation
    new_log_probs = log_prob(policy,states,actions)

    ratio = exp.(new_log_probs .- old_log_probs)
    π_loss = mean(ratio .* advantages)
    π_loss
end

function value_loss(policy,states::Array,returns::Array)
    return mean((policy.value_net(states) .- returns).^2)
end

function linesearch(policy,step_dir,states,actions,advantages,old_log_probs,kl_vars,old_params,num_steps=10;α=0.5)
    old_loss = policy_loss(policy,states,actions,advantages,old_log_probs).data

    for i in 1:num_steps
        # Obtain new parameters
        new_params = old_params .+ ((α^i) .* step_dir)

        # Set the new parameters to the policy
        set_flat_params(new_params,get_policy_net(policy))
	
        # Compute surrogate loss
        new_loss = policy_loss(policy,states,actions,advantages,old_log_probs).data
        
        # Compute kl divergence
        kl_div = kl_loss(policy,states,kl_vars).data
        
        # Output Statistics #
        # println("Old Loss : $old_loss")
        # println("New Loss : $new_loss")
        # println("KL Div : $kl_div")
        #####################
	
        if new_loss >= old_loss && (kl_div <= δ)
            println("---Success---")
            set_flat_params(new_params,get_policy_net(policy))
	    return nothing
        end
    end
    
    set_flat_params(old_params,get_policy_net(policy))
end

function trpo_update(policy,states,actions,advantages,returns,log_probs,kl_vars,old_params)
    model_params = get_policy_params(policy)
    policy_grads = Tracker.gradient(() -> policy_loss(policy,states,actions,advantages,log_probs),model_params)
    flat_policy_grads = get_flat_grads(policy_grads,get_policy_net(policy)).data

    x = conjugate_gradients(policy,states,kl_vars,Hvp,flat_policy_grads,10)
    println(minimum(x' * Hvp(policy,states,kl_vars,x)))
    
    step_dir = nothing
    try
    	step_dir = sqrt.((2 * δ) ./ (x' * Hvp(policy,states,kl_vars,x))) .* x
    catch
        println("Square root of a negative number received...Skipping update")
	return
    end
     
    # Do a line search and update the parameters
    linesearch(policy,step_dir,states,actions,advantages,log_probs,kl_vars,old_params)
    
    # Update value function
    for _ in 1:V_ITERS
    	value_params = get_value_params(policy)
    	gs = Tracker.gradient(() -> value_loss(policy,states,returns),value_params)
    	update!(opt_value,value_params,gs)
    end
end

function train_step() 
    clear(episode_buffer)
    collect_and_process_rollouts(policy,episode_buffer,EPISODE_LENGTH,stats_buffer)
    
    idxs = partition(shuffle(1:size(episode_buffer.exp_dict["states"])[end]),BATCH_SIZE)
    
    old_params = copy(get_flat_params(get_policy_net(policy)))

    for i in idxs
        # println("A")
        mb_states = episode_buffer.exp_dict["states"][:,i] 
        mb_actions = episode_buffer.exp_dict["actions"][:,i] 
        mb_advantages = episode_buffer.exp_dict["advantages"][:,i] 
        mb_returns = episode_buffer.exp_dict["returns"][:,i] 
        mb_log_probs = episode_buffer.exp_dict["log_probs"][:,i]
        mb_kl_vars = episode_buffer.exp_dict["kl_params"][i]
        
        trpo_update(policy,mb_states,mb_actions,mb_advantages,mb_returns,mb_log_probs,mb_kl_vars,old_params)
    end
end

function train()
    for i in 1:NUM_EPISODES
        println(i)
        train_step()
        println(mean(stats_buffer.exp_dict["rollout_returns"]))

        if i % SAVE_FREQUENCY == 0
            save_policy(policy,"../weights")
        end
    end
end

train()
