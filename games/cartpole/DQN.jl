using Flux
using Flux.Optimise
using Statistics: mean
using DataStructures: CircularBuffer
using Distributions: sample
using Printf
#using CuArrays

include("cartpole.jl")
# Load game environment
env = CartPoleEnv()
reset!(env)

# ----------------------------- Parameters -------------------------------------

STATE_SIZE = length(env.state)    # 4
ACTION_SIZE = length(env.action_space) # 2
MEM_SIZE = 100_000
BATCH_SIZE = 64
γ = 1.0f0   			  # discount rate

# Exploration params
ϵ = 1.0f0       # Initial exploration rate
ϵ_MIN = 1f-2    # Final exploratin rate
ϵ_DECAY = 995f-3

# Optimiser params
η = 1f-2   # Learning rate
η_decay = 1f-2

memory = CircularBuffer{Any}(MEM_SIZE) # Used to remember past results

# ------------------------------ Model Architecture ----------------------------

model = Chain(Dense(STATE_SIZE, 24, tanh),
              Dense(24, 48, tanh),
              Dense(48, ACTION_SIZE)) |> gpu

loss(x, y) = Flux.mse(model(x), y)

opt = Optimiser(ADAM(η), InvDecay(η_decay))

# ----------------------------- Helper Functions -------------------------------

get_ϵ(e) = max(ϵ_MIN, min(ϵ, 1.0f0 - log10(e * ϵ_DECAY)))

remember(state, action, reward, next_state, done) =
  push!(memory, (state, action, reward, next_state, done))

function action(state, train=true)
  train && rand() <= get_ϵ(e) && (return rand(1:ACTION_SIZE))
  act_values = model(state |> gpu)
  return Flux.onecold(act_values)
end

function replay()
  global ϵ
  batch_size = min(BATCH_SIZE, length(memory))
  minibatch = sample(memory, batch_size, replace = false)

  x = Matrix{Float32}(undef, STATE_SIZE, batch_size)
  y = Matrix{Float32}(undef, ACTION_SIZE, batch_size)
  for (iter, (state, action, reward, next_state, done)) in enumerate(minibatch)
    target = reward
    if !done
      target += γ * maximum(model(next_state |> gpu).data)
    end

    target_f = model(state |> gpu).data
    target_f[action] = target

    x[:, iter] .= state
    y[:, iter] .= target_f
  end
  x = x |> gpu
  y = y |> gpu

  Flux.train!(loss, params(model), [(x, y)], opt)

  ϵ *= ϵ > ϵ_MIN ? ϵ_DECAY : 1.0f0
end

step_action(a) = a == 1 ? 1 : -1

function episode!(env, train=true)
  done = false
  total_reward = 0f0
  while !done
    #render(env)
    s = env.state
    a = action(s, train)
    s′, r, done, _ = step!(env, step_action(a))
    total_reward += r
    train && remember(s, a, r, s′, done)
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
  print("Episode: $e | Score: $total_reward ")
  last_100_mean = mean(scores)
  print("Last 100 episodes mean score: $(@sprintf "%6.2f" last_100_mean)")
  if last_100_mean > 195
    println("\nCartPole-v0 solved!")
    break
  end
  println()
  replay()
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
