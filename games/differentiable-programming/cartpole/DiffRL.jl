# This implementation of DQN on cartpole is to verify the cartpole.jl env
using Flux, Gym, Printf
using Flux.Tracker: track, @grad, data#, gradient
using Flux.Optimise: Optimiser, _update_params!#, update!
using Statistics: mean
using DataStructures: CircularBuffer
using CuArrays

#Load game environment
env = CartPoleEnv()
reset!(env)

#ctx = Ctx(env)

#display(ctx.s)
#using Blink# when not on Juno
#body!(Blink.Window(), ctx.s)

# ----------------------------- Parameters -------------------------------------

STATE_SIZE = length(env.state)
ACTION_SIZE = length(env.action_space)
MAX_TRAIN_REWARD = env.x_threshold*env.θ_threshold_radians
SEQ_LEN = 8

# Optimiser params
η = 3f-2

# ------------------------------ Model Architecture ----------------------------

sign(x::TrackedArray) = track(sign, x)
@grad sign(x) = Base.sign.(data(x)), x̄ -> (x̄,)

model = Chain(Dense(STATE_SIZE, 24, relu),
              Dense(24, 48, relu),
              Dense(48, 1, tanh), x->sign(x)) |> gpu

opt = ADAM(η)

action(state) = model(state)

function loss(rewards)
  ep_len = size(rewards, 1)
  max_rewards = ones(Float32, ep_len) * MAX_TRAIN_REWARD |> gpu
  Flux.mse(rewards, max_rewards)
end

# ----------------------------- Helper Functions -------------------------------

function train_reward()
  state = env.state
  x, ẋ, θ, θ̇  = state[1:1], state[2:2], state[3:3], state[4:4]
  # Custom reward for training
  # Product of Triangular function over x-axis and θ-axis
  # Min reward = 0, Max reward = env.x_threshold * env.θ_threshold_radians
  x_upper = env.x_threshold .- x
  x_lower = env.x_threshold .+ x

  r_x     = max.(0f0, min.(x_upper, x_lower))

  θ_upper = env.θ_threshold_radians .- θ
  θ_lower = env.θ_threshold_radians .+ θ

  r_θ     = max.(0f0, min.(θ_upper, θ_lower))

  return r_x .* r_θ
end


function replay(rewards)
  #grads = gradient(() -> loss(rewards), params(model))
  #for p in params(model)
  #  update!(opt, p, grads[p])
  #end
  
  Flux.back!(loss(rewards))
  _update_params!(opt, params(model))
end

#@which _update_params!(opt, params(model))

function episode!(env, train=true)
  done = false
  total_reward = 0f0
  rewards = []
  frames = 1
  while !done && frames <= 200
    #render(env, ctx)
    #sleep(0.01)

    a = action(env.state)
    s′, r, done, _ = step!(env, a)
    total_reward += r
    
    train && push!(rewards, train_reward())
    
    if train && (frames % SEQ_LEN == 0 || done)
      rewards = vcat(rewards...)
      replay(rewards)
      rewards = []
      env.state = param(env.state.data)
    end
    
    frames += 1
  end
  total_reward
end

# -------------------------------- Testing -------------------------------------

function test()
  score_mean = 0f0
  for _=1:100
    reset!(env)
    total_reward = episode!(env, false)
    score_mean += total_reward / 100
  end
  return score_mean
end

# ------------------------------ Training --------------------------------------

e = 1

while true
  global e
  reset!(env)
  total_reward = episode!(env)
  print("Episode: $e | Score: $total_reward | ")

  score_mean = test()
  score_mean_str = @sprintf "%6.2f" score_mean
  print("Mean score over 100 test episodes: " * score_mean_str)

  println()

  if score_mean > 195
    println("CartPole-v0 solved!")
    break
  end
  e += 1
end
