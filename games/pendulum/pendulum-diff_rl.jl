using Flux, Gym
using Flux.Optimise: _update_params!
using Statistics: mean
using DataStructures: CircularBuffer
#using CuArrays

#Load game environment

env = PendulumEnv()
reset!(env)

# ----------------------------- Parameters -------------------------------------

STATE_SIZE = length(env.state)
ACTION_SIZE = 1#length(env.actions)
ACTION_BOUND = 2#env.action_space.hi
MAX_EP = 15_000
MAX_EP_LENGTH = 1000
SEQ_LEN = 25

# ------------------------------ Model Architecture ----------------------------

model = Chain(Dense(2, 24, relu),
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

  for ep=1:MAX_EP_LENGTH
    s = env.state
    a = model(s)
    s′, r, done, _ = step!(env, a)
    total_reward += r.data[1]
    if train
      push!(rewards, r)
      if ep == MAX_EP_LENGTH || ep % SEQ_LEN == 0 
        rewards = vcat(rewards...)
        update(rewards)
	rewards = []
        env.state = param(env.state.data)
      end
    end
  end

  total_reward
end

# ------------------------------ Training --------------------------------------

scores = CircularBuffer{Float32}(100)
for e=1:MAX_EP
  reset!(env)
  total_reward = episode!(env)
  push!(scores, total_reward)
  print("Episode: $e | Score: $total_reward ")
  last_100_mean = mean(scores)
  println("Last 100 episodes mean score: $last_100_mean")
end

# -------------------------------- Testing -------------------------------------

for e=1:MAX_EP
  reset!(env)
  total_reward = episode!(env, false)
  println("Episode: $e | Score: $total_reward")
end
