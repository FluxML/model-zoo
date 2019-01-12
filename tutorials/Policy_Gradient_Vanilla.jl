using Flux, CuArrays
using OpenAIGym
import Reinforce.action
import Flux.params
using CUDAnative: log
using Flux: onehot
using Statistics

#-------Define Custom Policy------------
mutable struct CartPolePolicy <: Reinforce.AbstractPolicy
  train::Bool

  function CartPolePolicy(train = true)
    new(train)
  end
end

#----------Setup Environment-------
env = GymEnv("CartPole-v0")

#----------Define Hyperparameters-------
STATE_SIZE = length(env.state)
ACTION_SIZE = length(env.actions)
γ = 1.00f0    # discount rate

η = 1e-3 # Learning rate

#---------_Define Model And Loss Function---------
policy = Chain(Dense(STATE_SIZE,30,relu),Dense(30,50,relu),Dense(50,ACTION_SIZE))

function loss(s,actions,A_t)
    pi = softmax(policy(s))
    logpi = log.(mean(pi .* actions,dims=1) .+ 1f-10)
    return mean(-logpi .* A_t)
end

opt = ADAM(params(policy),η)

println("Model Setup")

#-------Overriden action Method--------
function action(pi::CartPolePolicy,reward,state,action)
    state= Array(state)
    action_probs = softmax(policy(state))
    action = sample(1:ACTION_SIZE,Weights(action_probs)) - 1
    return action
end

#-------Overriden episode Method--------
function episode!(env,pi = RandomPolicy())
    ep = Episode(env,pi)

    experience = []
    for (s,a,r,s_) in ep
        push!(experience,(s,a,r,s_))
    end
    
    return experience
end

function train(env)
    experience = episode!(env,CartPolePolicy())
    states = Matrix{Float32}(undef,STATE_SIZE,length(experience))
    actions = Matrix{Float32}(undef,ACTION_SIZE,length(experience))    
    l = 0.0
    G_t = 0.0

    for (i,(s,a,r,s_)) in enumerate(reverse(experience))
        state = Array(s)
        act = zeros(ACTION_SIZE,1)
        act[a+1,1] = 1
         
        G_t = γ*G_t + r
    
        l = l .+ loss(state,act,G_t)   
        Flux.back!(loss(state,act,G_t))
        opt()
    end 
    l = l./(1.0*length(experience))
    return G_t,l
end

println("Setup Training")

NUM_EPISODES = 1000

for i in 1:NUM_EPISODES
    println("---")
    reset!(env)
    returns,loss_val = train(env)
    println("Returns : $returns")
    println("Loss : $loss_val")
end

