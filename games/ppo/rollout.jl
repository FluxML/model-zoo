"""
Implements rollout utility functions to collect trajectory experiences
"""

"""
Returns an episode's worth of experience
"""

function run_episode(env::Gym.EnvWrapper,policy,num_steps::Int)
    experience = []
    
    s = reset!(env)
    for i in 1:num_steps
        a = action(policy,s)

        s_,r,_ = step!(env,a)
        push!(experience,(s,a,r,s_))
        s = s_
        if env.done
           break 
        end
    end
    experience
end

addprocs(num_processes) 

@everywhere function collect(policy,env,num_steps::Int)
    run_episode(env,policy,num_steps::Int)
end

@everywhere function episode(policy,num_steps::Int)
  env = make(policy.env_wrap.ENV_NAME,:rgb)
  env.max_episode_steps = num_steps
  return collect(policy,env,num_steps::Int)
end

function get_rollouts(policy,num_steps::Int)
    g = []
    for  w in workers()
      push!(g, episode(policy,num_steps))
    end

    fetch.(g)
end

"""
Process and extraction information from rollouts
"""

function collect_and_process_rollouts(policy,episode_buffer::Buffer,num_steps::Int,stats_buffer::Buffer)
    rollouts = get_rollouts(policy,num_steps)
    
    # Process the variables
    states = []
    actions = []
    rewards = []
    next_states = []
    advantages = []
    returns = []
    log_probs = []
    kl_params = []   

    # Logging statistics
    rollout_returns = []
    
    for ro in rollouts
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        
        for i in 1:length(ro)
             push!(episode_states,ro[i][1])
             push!(episode_actions,ro[i][2])
             push!(episode_rewards,ro[i][3])
             push!(episode_next_states,ro[i][4])

             if typeof(ro[i][2]) <: Int64
                push!(log_probs,log_prob(policy,reshape(ro[i][1],length(ro[i][1]),1),[ro[i][2]]).data)
             elseif typeof(ro[i][2]) <: Array
                push!(log_probs,log_prob(policy,reshape(ro[i][1],length(ro[i][1]),1),reshape(ro[i][2],length(ro[i][2]),1)).data)
             end

	     # Kl divergence variables
             if typeof(policy) <: CategoricalPolicy
                push!(kl_params,log_probs[end])
             elseif typeof(policy) <: DiagonalGaussianPolicy
                μ = policy.μ(reshape(ro[i][1],length(ro[i][1]),1)).data
                logΣ = policy.logΣ.data
                push!(kl_params,[μ,logΣ])
             end
        end
        
        episode_rewards = scale_rewards(policy.env_wrap,episode_rewards)

        episode_advantages = gae(policy,episode_states,episode_actions,episode_rewards,episode_states,num_steps)
	episode_advantages = normalise(episode_advantages)

	if terminate_horizon == false
		    # println("Appending value of last state to returns")
	        episode_returns = disconunted_returns(episode_rewards,policy.value_net(episode_states[end]).data[1])
	else
		episode_returns = disconunted_returns(episode_rewards)
	end
	
        push!(states,hcat(episode_states...))
        push!(actions,hcat(episode_actions...))
        push!(rewards,hcat(episode_rewards...))
        push!(advantages,hcat(episode_advantages...))
        push!(returns,hcat(episode_returns...))
        
        # Variables for logging
        push!(rollout_returns,episode_returns)

    end
    
    # Normalize advantage across all processes
    # advantages = normalise_across_procs(hcat(advantages...))

    episode_buffer.exp_dict["states"] = hcat(states...)
    episode_buffer.exp_dict["actions"] = hcat(actions...)
    episode_buffer.exp_dict["rewards"] = hcat(rewards...)
    episode_buffer.exp_dict["advantages"] = hcat(advantages...) # hcat(advantages...)
    episode_buffer.exp_dict["returns"] = hcat(returns...)
    episode_buffer.exp_dict["log_probs"] = hcat(log_probs...)
    episode_buffer.exp_dict["kl_params"] = copy(kl_params)

    # Log the statistics
    add(stats_buffer,"rollout_rewards",sum(hcat(rewards...)))

    println("Rollout rewards : $(sum(hcat(rewards...)))")

end
