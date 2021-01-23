using Flux, Gym, Printf, Zygote
using Flux.Optimise: update!
using Statistics: mean
#using CuArrays

#Load game environment

env = make("Pendulum-v0")
reset!(env)
# ----------------------------- Parameters -------------------------------------


STATE_SIZE = length(state(env)) # returns state from obs space
ACTION_SIZE = 1#length(env.actions)
ACTION_BOUND = env._env.action_space.high[1]
MAX_REWARD = 0f0 # Max reward in a timestep
MAX_EP = 10
MAX_EP_LENGTH = 1000
SEQ_LEN = 4

# ------------------------------ Model Architecture ----------------------------

model = Chain(Dense(STATE_SIZE, 24, relu),
              Dense(24, 48, relu),
              Dense(48, ACTION_SIZE)) |> gpu

η = 3f-2

opt = ADAM(η)

loss(r) = Flux.mse(r, MAX_REWARD)

# ----------------------------- Helper Functions -------------------------------

function μEpisode(env::EnvWrapper)
    l = 0
    for frames ∈ 1:SEQ_LEN
        #render(env, ctx)
        #sleep(0.01)
        a = model(state(env))
        s, r, done, _ = step!(env, a)
        if trainable(env)
            l += loss(r)
        end

        game_over(env) && break
    end
    return l
end


function episode!(env::EnvWrapper)
  reset!(env)
  while !game_over(env)
    if trainable(env)
      grads = gradient(()->μEpisode(env), params(model))
      update!(opt, params(model), grads)
    else
      μEpisode(env)
    end
  end

  env.total_reward
end

# -------------------------------- Testing -------------------------------------

function test(env::EnvWrapper)
  score_mean = 0f0
  testmode!(env)
  for e=1:100
    total_reward = episode!(env)
    score_mean += total_reward / 100
  end
  testmode!(env, false)
  return score_mean
end

# ------------------------------ Training --------------------------------------

for e=1:MAX_EP
  total_reward = episode!(env)
  total_reward = @sprintf "%9.3f" total_reward
  print("Episode: $e | Score: $total_reward | ")
  score_mean = test(env)
  score_mean = @sprintf "%9.3f" score_mean
  println("Mean score over 100 test episodes: $score_mean")
end
