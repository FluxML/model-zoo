using Reinforce:CartPole, actions, reset!, Episode, finished
import Reinforce.action
using Flux, StatsBase, Plots
using Flux: back!, onehotbatch

gr()

#Define custom policy for choosing action
mutable struct CartPolePolicy <: Reinforce.AbstractPolicy end

#Load game environment
env = CartPole()

# ------------------------ Parameters -----------------------------------------

EPISODES = 3000
STATE_SIZE = length(env.state)
ACTION_SIZE = length(actions(env, env.state))
η = 5e-3   #learning rate

N_STEP_RETURN = 8
γ = 0.99
γᴺ = γ ^ N_STEP_RETURN

ϵ_START = 0.4  # exploration rate
ϵ_STOP = 0.15
ϵ_STEPS = 75000

cᵥ = 0.5			# v loss coefficient
cₑ = 0.01 # entropy coefficient

memory = []
MEM_SIZE = 2000

frames = 0

# ----------------------------- Model Architecture -----------------------------

base = Chain(Dense(4, 24, relu), Dense(24, 24, relu))
value = Dense(24, 1)
policy = Dense(24, ACTION_SIZE)

# ----------------------------- Loss ------------------------------------------

# Policy Loss
function loss_π(π, v, action, rₜ)
    logπ = log.(mean(π .* action, 1) + 1e-10)
    advantage = rₜ - v
    -logπ .* advantage.data #to stop backpropagation through advantage
end

# Value loss
lossᵥ(v, rₜ) = cᵥ * (rₜ - v) .^ 2

entropy(π) = cₑ * mean(π .* log.(π + 1e-10), 1)

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
  rₜ = r + γᴺ .* v′ .* s_mask	# set v to 0 where s_ is terminal state

  return mean(loss_π(π, v, a, rₜ) + lossᵥ(v, rₜ) + entropy(π))
end

# --------------------------- Training ----------------------------------------

opt = RMSProp(params(base) ∪ params(value) ∪ params(policy), η)

function train()
	x = hcat(memory...)
  back!(loss(x))
  opt()
end

# --------------------------- Helper Functions --------------------------------

function get_ϵ()
  if frames >= ϵ_STEPS
    return ϵ_STOP
  else
    return ϵ_START + frames * (ϵ_STOP - ϵ_START) / ϵ_STEPS	# linearly interpolate
  end
end

# stores the tuple of state, action, reward, next_state, and done
function remember(state, action, reward, next_state, done)
  if length(memory) == MEM_SIZE
    deleteat!(memory, 1)
  end
  push!(memory, [state, action, reward, next_state, done])
end

# Choose action according to policy CartPolePolicy
function action(p::CartPolePolicy, reward, state, action)
  ϵ = get_ϵ()
  if rand() <= ϵ
      return rand(1:ACTION_SIZE)
  end

  π = policy(base(state))
  sample(1:ACTION_SIZE, Weights(π.data)) # returns action
end


#Render the environment
on_step(env::CartPole, niter, sars) = gui(plot(env))

function episode!(env, p = RandomPolicy(); stepfunc = on_step, kw...)
    ep = Episode(env, p) # Runs an episode with policy p
    global frames
    for sars in ep
        stepfunc(ep.env, ep.niter, sars)
        state, action, reward, next_state = sars
        done = finished(ep.env, next_state) #check if game is over
        reward = !done ? reward : -1 #Penalty of -1 if game is over
        remember(state, action, reward, next_state, done)
        frames += 1
    end
    ep.total_reward
end

for e=1:EPISODES
    reset!(env)
    total_reward = episode!(env, CartPolePolicy())
    println("Episode: $e/$EPISODES | Score: $total_reward | ϵ: $(get_ϵ())")
    train()
end
