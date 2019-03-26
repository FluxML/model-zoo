using Flux, CuArrays
using OpenAIGym
import Reinforce.action
import Flux.params
using Flux.Tracker: grad, update!
using Flux: onehot
using Statistics

#-------Define Custom Policy------------
mutable struct CartPolePolicy <: Reinforce.AbstractPolicy
  train::Bool

  function CartPolePolicy(train = true)
    new(train)
  end
end

# #----------Setup Environment-------
env = GymEnv("CartPole-v0")

# #----------Define Hyperparameters-------
STATE_SIZE = length(env.state)
ACTION_SIZE = length(env.actions)
γ = 1.0f0    # discount rate

η = 1e-4 # Learning rate

#---------_Define Model And Loss Function---------
policy = Chain(Dense(STATE_SIZE,30,relu),Dense(30,50,relu),Dense(50,ACTION_SIZE)) |> gpu

function loss(s,actions,A_t)
    s = s |> gpu
    actions = actions |> gpu
    pi_ = softmax(policy(s)) |> gpu
    A_t = A_t |> gpu
    logpi = log.(mean(pi_ .* actions) .+ 1f-10)
    return mean(-logpi .* A_t)
end

opt = ADAM(η)


println("Model Setup")

function discount_and_normalize_returns(experience)
    returns = []
    G_t = 0.0

    for (i,(s,a,r,s_)) in enumerate(experience)
        G_t = G_t*γ + r
        push!(returns,G_t)
    end

    μ = mean(returns)
    σ² = var(returns)

    dis_normalized_ret = [(G_t - μ)/(σ² + 1f-10) for G_t in returns]
    return dis_normalized_ret
end

#-------Overriden action Method--------
function action(pi::CartPolePolicy,reward,state,action)
    state= Array(state) |> gpu
    action_probs = softmax(policy(state))
    action = sample(1:ACTION_SIZE,Weights(action_probs)) - 1
    return action
end

#-------Overriden episode Method--------
function episode!(env,pi = RandomPolicy())
    ep = Episode(env,pi)

    experience = []
    for (s,a,r,s_) in ep
	# OpenAIGym.render(env)
        push!(experience,(s,a,r,s_))
    end
    
    return experience
end

function train(env)
    experience = episode!(env,CartPolePolicy())
    l = 0.0
    G_t = 0.0

    ep_r = 0.0

    dis_normalized_ret = discount_and_normalize_returns(experience)

    for (i,(s,a,r,s_)) in enumerate(reverse(experience))
        state = Array(s)
        act = zeros(ACTION_SIZE,1)
        act[a+1,1] = 1
        
        ep_r = ep_r + r
        G_t = dis_normalized_ret[i]
        l = l + loss(state,act,G_t)   
        Flux.back!(loss(state,act,G_t))
        update!(opt,params(policy),Tracker.gradient(() -> loss(state,act,G_t), params(policy)))
    end 
    l = l./(1.0*length(experience))
    return ep_r,l,length(experience)
end

println("Setup Training")

NUM_EPISODES = 5000

for i in 1:NUM_EPISODES
    println("---Episode : $i---")
    reset!(env)
    returns,loss_val,len = train(env)
    println("Rewards : $returns")
    println("Loss : $loss_val")
    println("Episode Length : $len")
end