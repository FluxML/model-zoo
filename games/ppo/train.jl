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

println("imported")

num_processes = 1
include("../common/policies.jl")
include("../common/utils.jl")
include("../common/buffer.jl")
include("rollout.jl")

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
    register(sb,"rollout_rewards")
    
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

#----------------Hyperparameters----------------#
# Environment Variables #
ENV_NAME = "Pendulum-v0"
EPISODE_LENGTH = 2000
terminate_horizon = false
resume = false
# Policy parameters #
η = 1e-3 # Learning rate
STD = 0.0 # Standard deviation
# GAE parameters
γ = 0.99
λ = 0.95
# Optimization parameters
PPO_EPOCHS = 10
NUM_EPISODES = 100000
BATCH_SIZE = EPISODE_LENGTH
c₀ = 1.0
c₁ = 1.0
c₂ = 0.01
# PPO parameters
ϵ = 0.1
# FREQUENCIES
SAVE_FREQUENCY = 10 
VERBOSE_FREQUENCY = 5
global_step = 0

# Define policy
env_wrap = EnvWrap(ENV_NAME)

if resume == true
    policy = load_policy(env_wrap,"./weights/")
else
    policy = get_policy(env_wrap)
end

println(policy)

# Define buffers
episode_buffer = initialize_episode_buffer()
stats_buffer = initialize_stats_buffer()

# Define optimizer
opt = ADAM(η)

function loss(policy,states::Array,actions::Array,advantages::Array,returns::Array,old_log_probs::Array)
    new_log_probs = log_prob(policy,states,actions)
    
    # Surrogate loss computations
    ratio = exp.(new_log_probs .- old_log_probs)
    surr1 = ratio .* advantages
    surr2 = clamp.(ratio,(1.0 - ϵ),(1.0 + ϵ)) .* advantages
    policy_loss = mean(min.(surr1,surr2))
    
    value_predicted = policy.value_net(states)
    value_loss = mean((value_predicted .- returns).^2)
    
    entropy_loss = mean(entropy(policy,states))
    
    -c₀*policy_loss + c₁*value_loss - c₂*entropy_loss
end

function ppo_update(policy,states::Array,actions::Array,advantages::Array,returns::Array,old_log_probs::Array,kl_vars)
    model_params = params(get_policy_params(policy)...,get_value_params(policy)...)
    
    # Calculate gradients
    gs = Tracker.gradient(() -> loss(policy,states,actions,advantages,returns,old_log_probs),model_params)
    
    # Take a step of optimisation
    update!(opt,model_params,gs)
end

function train_step()    
    clear(episode_buffer)
    collect_and_process_rollouts(policy,episode_buffer,EPISODE_LENGTH,stats_buffer)
    
    idxs = partition(1:size(episode_buffer.exp_dict["states"])[end],BATCH_SIZE)
    
    early_stop = false
    for epoch in 1:PPO_EPOCHS
        for i in idxs
            mb_states = episode_buffer.exp_dict["states"][:,i] 
            mb_actions = episode_buffer.exp_dict["actions"][:,i] 
            mb_advantages = episode_buffer.exp_dict["advantages"][:,i] 
            mb_returns = episode_buffer.exp_dict["returns"][:,i] 
            mb_log_probs = episode_buffer.exp_dict["log_probs"][:,i]
	    mb_kl_vars = episode_buffer.exp_dict["kl_params"][i]
	
	    
    	    kl_div = mean(kl_divergence(policy,mb_kl_vars,mb_states))
    	    println("KL Sample : $(kl_div)")

            ppo_update(policy,mb_states,mb_actions,mb_advantages,mb_returns,mb_log_probs,mb_kl_vars)
        end
	
	if early_stop == true
		break
	end
    end
end

function train()
    for i in 1:NUM_EPISODES
        println(i)
        train_step()
        println(mean(stats_buffer.exp_dict["rollout_rewards"]))

        if i % SAVE_FREQUENCY == 0
            save_policy(policy,"./weights/")
        end
    end
end

train()
