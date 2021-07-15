using Reinforce:Pendulum, actions, reset!, Episode, finished, step!, state
import DataStructures
using Reinforce
import Reinforce.action
using Flux, StatsBase, Plots, Statistics
using Flux: back!, onehotbatch

gr()

#Load game environment
env = Pendulum()

# ---------------------------------- Parameters ---------------------------------------#

EPISODES = 15000
STATE_SIZE = length(state(env))
ACTION_SIZE = length(actions(env, state(env)))
η = 5e-3   #learning rate

γ = 0.99

ϵ_START = 0.4  # exploration rate
ϵ_STOP = 0.15
ϵ_STEPS = 75000

cᵥ = 0.5			# v loss coefficient
cₑ = 0.01 # entropy coefficient

memory = DataStructures.CircularBuffer{Any}(13000)

frames = 0

# ----------------------------- Model Architecture ---------------------------------#

base = Chain(Dense(STATE_SIZE, 24, relu), Dense(24, 24, relu))
value = Dense(24, 1)
policy = Dense(24, ACTION_SIZE)

# ----------------------------- Loss --------------------------------#

# Policy Loss
function loss_π(π, v, action, rₜ)
    logπ = log.(mean(π .* action, dims=1) .+ 1e-10)
    advantage = rₜ - v
    -logπ .* advantage.data          #to stop backpropagation through advantage
end

# Value loss
lossᵥ(v, rₜ) = cᵥ * (rₜ - v) .^ 2

entropy(π) = cₑ * mean(π .* log.(π .+ 1e-10), dims=1)

# Total Loss = Policy loss + Value Loss + Entropy
function loss(x)
  s = hcat(x[1, :]...)
  a = onehotbatch(x[2, :], 1:ACTION_SIZE)
  r = hcat(x[3, :]...)
  s′ = hcat(x[4, :]...)
  s_mask = .!hcat(x[5, :]...)

  base_out = base(s)
  v = value(base_out)
  π = softmax(policy(base_out))

  v′ = value(base(s′))
  rₜ = r + γ .* v′ .* s_mask	# set v to 0 where s_ is terminal state

  mean(loss_π(π, v, a, rₜ) + lossᵥ(v, rₜ) + entropy(π))
end

# --------------------------- Training ----------------------------#

opt() = RMSProp(η)

function train()
  x = hcat(memory...)
  back!(loss(x))
  opt()
end

# --------------------------- Helper Functions --------------------------------#

function get_ϵ()
  if frames >= ϵ_STEPS
    return ϵ_STOP
  else
    return ϵ_START + frames * (ϵ_STOP - ϵ_START) / ϵ_STEPS	# linearly interpolate
  end
end

# stores the tuple of state, action, reward, next_state, and done
function remember(state, action, reward, next_state, done)
  push!(memory, [state, action, reward, next_state, done])
end

# Choose action according to policy CartPolePolicy
function act(state)
  ϵ = get_ϵ()
  if rand() <= ϵ
      return rand(1:ACTION_SIZE)
  end

  π = policy(base(state))
  sample(1:ACTION_SIZE, Weights(vec(π.data))) # returns action
end


#Render the environment
#gui(plot(env))

#-------------------------------Testing--------------------------------#
for e = 1:EPISODES
    Reinforce.reset!(env)
    sta = state(env)
    global frames
    sta = reshape(sta, STATE_SIZE, 1)
    total_reward = 0
    while true
      #gui(plot(env))
      action = act(sta)
      reward, next_state = step!(env, sta, action)
      done = finished(env, next_state)
      reward = !done ? reward : -reward
      total_reward += reward
      next_state = reshape(next_state, STATE_SIZE, 1)
      remember(sta, action, reward, next_state, done)
      frames += frames == 75000 ? 0 : 1
      sta = next_state
      if done
        println("Episode: $e | Score: $total_reward | ϵ: $(get_ϵ())")
        break
      end
    end
    train()
end