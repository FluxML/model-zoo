using Flux
using Flux:params
using OpenAIGym
import Reinforce:action

# ------------------------ Load game environment -------------------------------
env = GymEnv("Pong-v0")

# Custom Policy for Pong-v0
mutable struct PongPolicy <: Reinforce.AbstractPolicy
  prev_state
  train::Bool
  function PongPolicy(train = true)
    new(zeros(STATE_SIZE), train)
  end
end

# ---------------------------- Parameters --------------------------------------
EPISODES = 1000
STATE_SIZE = 80 * 80 # after preprocessing
ACTION_SPACE = 2 # considering only up and down
MEM_SIZE = 10000 # Size of replay buffer
BATCH_SIZE = 32 # Size of batch for replay
γ = 0.95    # discount rate
ϵ_START = 1.0  # exploration
ϵ_STOP = 0.05
ϵ_STEPS = 10000
η = 2.5e-4   #learning rate
N_TARGET = 200 # Update target model every N_TARGET steps

timesteps = 0
frames = 0

memory = [] #used to remember past results

# --------------------------- Model Architecture -------------------------------

value = Dense(200, 1, relu)
adv = Dense(200, ACTION_SPACE, relu)

Q(x::TrackedArray) = broadcast(+, value(x), broadcast(+, adv(x), -mean(adv(x), 1)))
model = Chain(Dense(STATE_SIZE, 200, relu), x -> Q(x))

model_target = deepcopy(model)

huber_loss(x, y) = mean(sqrt.(1 + (model(x) - y) .^ 2) - 1)

opt() = RMSProp(params(model), η)

fit_model(data) = Flux.train!(huber_loss, data, opt)

# ------------------------------- Helper Functions -----------------------------

get_ϵ() = frames == ϵ_STEPS ? ϵ_STOP : ϵ_START + frames * (ϵ_STOP - ϵ_START) / ϵ_STEPS

function preprocess(I)
  #= preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector =#
  I = I[36:195, :, :] # crop
  I = I[1:2:end, 1:2:end, 1] # downsample by factor of 2
  I[I .== 144] = 0 # erase background (background type 1)
  I[I .== 109] = 0 # erase background (background type 2)
  I[I .!= 0] = 1 # everything else (paddles, ball) just set to 1

  return I[:] #Flatten and return
end

# Putting data into replay buffer
function remember(prev_s, s, a, r, s′, done)
  if length(memory) == MEM_SIZE
    deleteat!(memory, 1)
  end

  state = preprocess(s) - prev_s
  next_state = env.done ? zeros(STATE_SIZE) : preprocess(s′)
  next_state -= preprocess(s)
  reward = σ(r)
  push!(memory, (state, a, r, next_state, done))
end

function action(π::PongPolicy, reward, state, action)
  if rand() <= get_ϵ() && π.train
    return rand(1:ACTION_SPACE) + 1 # UP and DOWN action corresponds to 2 and 3
  end

  s = preprocess(state) - π.prev_state
  act_values = model(s)
  return Flux.argmax(act_values)[1] + 1  # returns action max Q-value
end

function replay()
  global ϵ, timesteps, model_target, frames

  minibatch = sample(memory, BATCH_SIZE, replace = false)

  for (state, action, reward, next_state, done) in minibatch
    target = reward

    if !done
      a_max = Flux.argmax(model(next_state))
      target += γ * model_target(next_state).data[a_max]
    end

    target_f = model(state).data
    target_f[action] = target
    dataset = zip(state, target_f)
    fit_model(dataset)

    timesteps = (timesteps + 1) % N_TARGET
    frames += frames < ϵ_STEPS ? 1 : 0

    # Update target model
    if timesteps == 0
      model_target = deepcopy(model)
    end
  end
end

function episode!(env, π = RandomPolicy())
  ep = Episode(env, π)

  for (s, a, r, s′) in ep
    OpenAIGym.render(env)
    if π.train remember(π.prev_state, s, a - 1, r, s′, env.done) end
    π.prev_state = preprocess(s)
  end

  ep.total_reward
end

# ------------------------------ Training --------------------------------------

e = 1
while e < EPISODES
  reset!(env)
  total_reward = episode!(env, PongPolicy())
  println("Episode: $e | Score: $total_reward | ϵ: $(get_ϵ())")
  if length(memory) >= BATCH_SIZE
    replay()
  end
  e += 1
end

# -------------------------------- Testing -------------------------------------
ee = 1

while true
  reset!(env)
  total_reward = episode!(env, PongPolicy(false))
  println("Episode: $ee | Score: $total_reward")
  ee += 1
end
