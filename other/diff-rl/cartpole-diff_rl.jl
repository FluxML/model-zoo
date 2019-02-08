# This implementation of DQN on cartpole is to verify the cartpole.jl env
using Flux
using Flux: Tracker
using Flux.Optimise: Optimiser, _update_params!
using Statistics: mean
using DataStructures: CircularBuffer
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

# Optimiser params
η = 1f-2  # Learning rate

# ------------------------------ Model Architecture ----------------------------

model = Chain(Dense(STATE_SIZE, 24, relu),
              Dense(24, 48, relu),
              Dense(48, 1, tanh)) |> gpu

loss(r) = Flux.mse(r, env.x_threshold*env.θ_threshold_radians)

opt = ADAM(η)

# ----------------------------- Helper Functions -------------------------------

function replay(a, r, action)
  Flux.back!(loss(r))
  grads = [Tracker.grad(action)] |> gpu
  Flux.back!(a, grads)
  _update_params!(opt, params(model))
end

function episode!(env, train=true)
  done = false
  total_reward = 0f0
  while !done
    #render(env, ctx)
    #sleep(0.01)

    s = env.state
    a = model(s |> gpu)
    s′, r, done, train_reward, action = step!(env, sign(a.data[1]))
    total_reward += r
    train && replay(a, train_reward, action)
  end

  total_reward
end

# ------------------------------ Training --------------------------------------

e = 1
scores = CircularBuffer{Float32}(100)

while true
  global e
  reset!(env)
  total_reward = episode!(env)
  push!(scores, total_reward)
  print("Episode: $e | Score: $total_reward | ")
  last_100_mean = mean(scores)
  print("Last 100 episodes mean score: $last_100_mean")
  if last_100_mean > 195
    println("\nCartPole-v0 solved!")
    break
  end
  println()
  e += 1
end
# -------------------------------- Testing -------------------------------------
ee = 1


while true

  global ee
  reset!(env)
  total_reward = episode!(env, false)
  println("Episode: $ee | Score: $total_reward")
  ee += 1
end
