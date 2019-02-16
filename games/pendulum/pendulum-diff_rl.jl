using Flux, Gym, Printf
using Flux.Tracker: data
using Flux.Optimise: _update_params!
using Statistics: mean
using DataStructures: CircularBuffer
using CuArrays

#Load game environment

env = PendulumEnv()

# ----------------------------- Parameters -------------------------------------

STATE_SIZE = length(reset!(env)) # returns state from obs space
ACTION_SIZE = 1#length(env.actions)
ACTION_BOUND = 2#env.action_space.hi
MAX_EP = 15_000
MAX_EP_LENGTH = 1000
SEQ_LEN = 25

# ------------------------------ Model Architecture ----------------------------

model = Chain(Dense(STATE_SIZE, 24, relu),
              Dense(24, 48, relu),
              Dense(48, ACTION_SIZE)) |> gpu

η = 3f-2

opt = ADAM(η)

function loss(r)
  seq_len = size(r, 1)
  z = zeros(Float32, seq_len) |> gpu
  Flux.mse(r, z)
end

# ----------------------------- Helper Functions -------------------------------

function update(rewards)
  Flux.back!(loss(rewards))
  _update_params!(opt, params(model))
  # grads = Tracker.gradient(()->loss(rewards), params(model))
  # for p in params(model)
  #   update!(opt, p, grads[p])
  # end
end

function episode!(env, train=true)
  total_reward = 0f0
  rewards = []
  s = reset!(env)
  for ep=1:MAX_EP_LENGTH
    a = model(s)
    s′, r, done, _ = step!(env, a)
    total_reward += data(r)[1]
    s = s′
    if train
      push!(rewards, r)
      if ep == MAX_EP_LENGTH || ep % SEQ_LEN == 0 
        rewards = vcat(rewards...)
        update(rewards)
	rewards = []
        env.state = param(data(env.state))
        s = Gym._get_obs(env)
      end
    end
  end

  total_reward
end

# -------------------------------- Testing -------------------------------------

function test()
  score_mean = 0f0
  for e=1:100
    total_reward = episode!(env, false)
    score_mean += total_reward / 100
  end
  return score_mean
end

# ------------------------------ Training --------------------------------------

for e=1:MAX_EP
  total_reward = episode!(env)
  total_reward = @sprintf "%9.3f" total_reward
  print("Episode: $e | Score: $total_reward | ")
  score_mean = test()
  score_mean = @sprintf "%9.3f" score_mean
  println("Mean score over 100 test episodes: $score_mean")
end
