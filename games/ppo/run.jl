using Pkg
Pkg.activate(".")

using Flux
using Gym
using OpenAIGym
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

include("common/policies.jl")
include("common/utils.jl")

ENV_NAME = "CartPole-v0"
TEST_STEPS = 10000
global steps_run = 0

LOAD_PATH = "../weights/ppo/CartPole-v0/"

# Define policy
env_wrap = EnvWrap(ENV_NAME)

# env = make(ENV_NAME,:rgb)
# env.max_episode_steps = TEST_STEPS
env = GymEnv(ENV_NAME)
env.pyenv._max_episode_steps = TEST_STEPS
policy = load_policy(env_wrap,LOAD_PATH)

# Test Run Function
function test_run(env)
	global steps_run
 	# testmode!(env)
    ep_r = 0.0
    
    s = OpenAIGym.reset!(env)
    println(s)
    for i in 1:TEST_STEPS
	if i % 1000 == 0
		println("---Resetting---")
		s = OpenAIGym.reset!(env)
		println(s)
		ep_r = 0.0
	end

        OpenAIGym.render(env)
    a = test_action(policy,s)
    a = convert.(Float64,a)

        if typeof(policy) <: DiagonalGaussianPolicy
            a = reshape(a,env_wrap.ACTION_SIZE)
        else
            # OpanAIGym hack
            a = a - 1.0f0
            a = Int64.(a)
        end

	# a = action(policy,s)
        # s_,r,_ = step!(env,a)
        r,s_ = OpenAIGym.step!(env,a)
	println(a)
    sleep(0.01)

        ep_r += r
        
		steps_run += 1

        s = s_
        if env.done
           # break 
           println("---Resetting---")
           s = OpenAIGym.reset!(env)
           println(ep_r)
           ep_r = 0.0
           continue
        end
    end
    ep_r
end

total_reward = test_run(env)

println("TOTAL STEPS :: $steps_run :: TOTAL REWARD :: $total_reward")
