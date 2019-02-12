using Flux, Gym
using Flux.Tracker: data, grad
using Flux.Optimise: _update_params!
using Statistics: mean
using DataStructures: CircularBuffer
using Distributions: sample

using CuArrays

#Load game environment

env = PendulumEnv()
reset!(env)

# ----------------------------- Parameters -------------------------------------

STATE_SIZE = 3
ACTION_SIZE = 1#length(env.actions)
ACTION_BOUND = 2#env.action_space.hi
MAX_EP = 50_000
MAX_EP_LENGTH = 200

BATCH_SIZE = 64
MEM_SIZE = 100_000
γ = 99f-2     # discount rate
τ = 1f-2 # for running average while updating target networks
η_act = 1f-4   # Learning rate
η_crit = 1f-3

memory = CircularBuffer{Any}(MEM_SIZE)

# -------------------------------- Action Noise --------------------------------

mutable struct OUNoise
  mu
  theta
  sigma
  X
end

ou = OUNoise(0f0, 15f-2, 2f-1, [0f0])

function sample_noise(ou::OUNoise)
  dx = ou.theta * (ou.mu .- ou.X)
  dx = dx .+ ou.sigma * randn(Float32, length(ou.X))
  ou.X .+= + dx
end

# ----------------------------- Model Architecture -----------------------------
w_init(dims...) = 6f-3rand(Float32, dims...) .- 3f-3

actor = Chain(Dense(STATE_SIZE, 400), BatchNorm(400, relu),
	      	    Dense(400, 300), BatchNorm(300, relu),
              Dense(300, ACTION_SIZE, tanh, initW=w_init),
              x -> x * ACTION_BOUND) |> gpu
actor_target = deepcopy(actor)

mutable struct crit
  state_crit
  act_crit
  state_act_crit
end

Flux.@treelike crit

function (c::crit)(state, action)
  x = c.state_crit(state) .+ c.act_crit(action)
  c.state_act_crit(x)
end

critic = crit(Chain(Dense(STATE_SIZE, 400), BatchNorm(400, relu), Dense(400, 300)) |> gpu,
              Dense(ACTION_SIZE, 300) |> gpu,
              Dense(300, 1, initW=w_init) |> gpu
             )

Base.deepcopy(c::crit) = crit(deepcopy(c.state_crit),
                              deepcopy(c.act_crit),
                              deepcopy(c.state_act_crit))
critic_target = deepcopy(critic)

# ------------------------------- Param Update Functions---------------------------------

function update_target!(target, model; τ = 1f0)
  for (p_t, p_m) in zip(params(target), params(model))
    p_t.data .= (1f0 - τ) * p_t.data .+ τ * p_m.data
  end
end

nullify_grad!(p) = p
nullify_grad!(p::TrackedArray) = (p.grad .= 0f0)

zero_grad!(model) = (model = mapleaves(nullify_grad!, model))

# ---------------------------------- Training ----------------------------------

opt_crit = ADAM(η_crit)
opt_act  = ADAM(η_act)

function replay()
  # Getting data in shape
  minibatch = sample(memory, BATCH_SIZE)
  x = hcat(minibatch...)

  s      =   hcat(x[1, :]...) |> gpu
  a      =   hcat(x[2, :]...) |> gpu
  r      =   hcat(x[3, :]...) |> gpu
  s′     =   hcat(x[4, :]...) |> gpu
  s_mask = .!hcat(x[5, :]...) |> gpu

  # Update Critic
  a′ = data(actor_target(s′))
  v′ = data(critic_target(s′, a))
  y = r .+ γ * v′ .* s_mask	# set v′ to 0 where s_ is terminal state

  v = critic(s, a)
  loss_crit = Flux.mse(y, v)

  # Update Actor
  actions = actor(s)
  crit_in = (s, param(data(actions)))
  crit_out = critic(crit_in...)
  Flux.back!(sum(crit_out))

  act_grads = -grad(crit_in[2])
  zero_grad!(actor)
  Flux.back!(actions, act_grads)  # Chain rule
  _update_params!(opt_act, params(actor))

  zero_grad!(critic)
  Flux.back!(loss_crit)
  _update_params!(opt_crit, params(critic))
end

# --------------------------- Helper Functions --------------------------------

# stores the tuple of state, action, reward, next_state, and done
remember(state, action, reward, next_state, done) =
  push!(memory, [data.((state, action, reward, next_state))..., done])

# Choose action according to policy PendulumPolicy
function action(state, train=true)
  state = reshape(data(state), size(state)..., 1)
  act_pred = data(actor(state |> gpu)) .+  ACTION_BOUND * sample_noise(ou)[1] * train
  clamp(data(act_pred[1]), -ACTION_BOUND, ACTION_BOUND) # returns action
end

function episode!(env, train=true)
  total_reward = 0f0
  s = reset!(env)
  for ep=1:MAX_EP_LENGTH
    a = action(s, train)
    s′, r, done, _ = step!(env, a)
    remember(s, a, r, s′, ep==MAX_EP_LENGTH)
    total_reward += data(r)[1]

    if train && length(memory) ≥ BATCH_SIZE
      replay()
      update_target!(actor_target, actor; τ = τ)
      update_target!(critic_target, critic; τ = τ)
    end
  end
  total_reward
end

# ------------------------------ Training --------------------------------------
scores = CircularBuffer{Float32}(100)
for e=1:MAX_EP
  total_reward = episode!(env)
  push!(scores, total_reward)
  print("Episode: $e | Score: $total_reward | ")
  last_100_mean = mean(scores)
  println("Last 100 episodes mean score: $last_100_mean")
end


# -------------------------------- Testing -------------------------------------

for e=1:MAX_EP
  total_reward = episode!(env, false)
  println("Episode: $e | Score: $total_reward")
end
