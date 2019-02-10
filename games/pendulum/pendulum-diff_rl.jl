using Flux, Gym
using Flux.Optimise: _update_params!
using Statistics: mean
using DataStructures: CircularBuffer
using CuArrays

#Load game environment

env = PendulumEnv()
reset!(env)

# ----------------------------- Parameters -------------------------------------

STATE_SIZE = length(env.state)
ACTION_SIZE = 1#length(env.actions)
ACTION_BOUND = 2#env.action_space.hi
MAX_EP = 15_000
MAX_EP_LENGTH = 750

# ------------------------------ Model Architecture ----------------------------

model = Chain(Dense(2, 24, relu),
              Dense(24, 48, relu),
              Dense(48, ACTION_SIZE)) |> gpu

η = 3f-2

opt = ADAM(η)
# Max possible reward in a step in 0.
# Training one episode in a go
z = zeros(Float32, MAX_EP_LENGTH) |> gpu
loss(r) = Flux.mse(r, z)

# ----------------------------- Helper Functions -------------------------------

function update(rewards)
  Flux.back!(loss(rewards |> gpu))
  _update_params!(opt, params(model))
end

function episode!(env, train=true)
  total_reward = 0f0
  u = nothing
  rewards = []
  for _=1:MAX_EP_LENGTH
    s = env.state
    a = model(s |> gpu)
    s′, r, done, _ = step!(env, a)
    push!(rewards, r)
  end
  rewards = vcat(rewards...)
  train && update(rewards)
  total_reward = sum(Flux.Tracker.data(rewards))
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
