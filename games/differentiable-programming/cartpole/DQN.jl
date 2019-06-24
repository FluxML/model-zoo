using Flux, Gym, Printf, Zygote
using Statistics: mean
using DataStructures: CircularBuffer
using Distributions: sample
#using CuArrays

# Load game environment
env = make("CartPole-v0")
reset!(env)

# ----------------------------- Parameters -------------------------------------

STATE_SIZE = length(state(env))    # 4
ACTION_SIZE = length(env._env.action_space) # 2
MEM_SIZE = 100_000
BATCH_SIZE = 64
γ = 1f0   			  # discount rate

# Exploration params
ϵ = 1f0       # Initial exploration rate
ϵ_MIN = 1f-2    # Final exploratin rate
ϵ_DECAY = 995f-3

# Optimiser params
η = 1f-2   # Learning rate
η_decay = 1f-3

memory = CircularBuffer{Any}(MEM_SIZE) # Used to remember past results

# ------------------------------ Model Architecture ----------------------------

model = Chain(Dense(STATE_SIZE, 24, tanh),
              Dense(24, 48, tanh),
              Dense(48, ACTION_SIZE)) |> gpu

loss(x, y) = Flux.mse(model(x), y)

opt = Flux.Optimiser(ADAM(η), InvDecay(η_decay))

# ----------------------------- Helper Functions -------------------------------

get_ϵ(e) = max(ϵ_MIN, min(ϵ, 1f0 - log10(e * ϵ_DECAY)))

remember(state, action, reward, next_state, done) =
  push!(memory, (state, action, reward, next_state, done))

function action(state, train=true)
  train && rand() ≤ get_ϵ(e) && (return Gym.sample(env._env.action_space))
  act_values = model(state |> gpu)
  return Flux.onecold(act_values)
end

function replay()
  global ϵ
  batch_size = min(BATCH_SIZE, length(memory))
  minibatch = sample(memory, batch_size, replace = false)

  x = []
  y = []
  for (iter, (state, action, reward, next_state, done)) in enumerate(minibatch)
    target = reward
    if !done
      target += γ * maximum(model(next_state |> gpu))
    end

    target_f = model(state |> gpu)
    target_f[action] = target

    push!(x, state)
    push!(y, target_f)
  end
  x = hcat(x...) |> gpu
  y = hcat(y...) |> gpu

  grads = Zygote.gradient(()->loss(x, y), params(model))
  Flux.Optimise.update!(opt, params(model), grads)

  ϵ *= ϵ > ϵ_MIN ? ϵ_DECAY : 1.0f0
end

function episode!(env)
  reset!(env)
  while !game_over(env)
    #render(env)
    s = state(env)
    a = action(s, trainable(env))
    s′, r, done, _ = step!(env, a)
    trainable(env) && remember(s, a, r, s′, done)
  end

  env.total_reward
end

# -------------------------------- Testing -------------------------------------

function test(env::EnvWrapper)
  score_mean = 0f0
  testmode!(env)
  for _=1:100
      total_reward = episode!(env)
      score_mean += total_reward / 100
  end
  testmode!(env, false)
  return score_mean
end

# ------------------------------ Training --------------------------------------

e = 1
while true
    global e
    total_reward = @sprintf "%6.2f" episode!(env)
    print("Episode: $e | Score: $total_reward | ")
    replay()

    score_mean = test(env)
    score_mean_str = @sprintf "%6.2f" score_mean
    print("Mean score over 100 test episodes: " * score_mean_str)

    println()

    if score_mean > env.reward_threshold
        println("CartPole-v0 solved!")
    break
    end
    e += 1
end
