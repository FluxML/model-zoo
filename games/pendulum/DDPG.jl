using Flux, Gym, Printf
using Flux.Tracker: data, grad
using Flux.Optimise: _update_params!
using Statistics: mean
using DataStructures: CircularBuffer
using Distributions: sample

using CuArrays

#Load game environment

env = PendulumEnv()

# ----------------------------- Parameters -------------------------------------

STATE_SIZE = length(reset!(env))
ACTION_SIZE = 1#length(env.actions)
ACTION_BOUND = 2#env.action_space.hi
MAX_EP = 50_000
MAX_EP_LENGTH = 1000

BATCH_SIZE = 64
MEM_SIZE = 100_000
MIN_EXP_SIZE = 50_000

γ = 99f-2     # discount rate

τ = 1f-3 # for running average while updating target networks
η_act = 1f-4   # Learning rate
η_crit = 1f-3
L2_DECAY = 1f-2

# Ornstein-Uhlenbeck Noise params
μ = 0f0
θ = 15f-2
σ = 2f-1

# --------------------------------- Memory ------------------------------------

memory = CircularBuffer{Any}(MEM_SIZE)

function getData(batch_size = BATCH_SIZE)
  # Getting data in shape
  minibatch = sample(memory, batch_size)
  x = hcat(minibatch...)

  s      =   hcat(x[1, :]...) |> gpu
  a      =   hcat(x[2, :]...) |> gpu
  r      =   hcat(x[3, :]...) |> gpu
  s′     =   hcat(x[4, :]...) |> gpu
  s_mask = .!hcat(x[5, :]...) |> gpu

  return s, a, r, s′, s_mask
end

# -------------------------------- Action Noise --------------------------------

struct OUNoise
  μ
  θ  
  σ
  X
end

ou = OUNoise(μ, θ, σ, zeros(Float32, ACTION_SIZE) |> gpu)

function sample_noise(ou::OUNoise)
  dx     = ou.θ * (ou.μ .- ou.X)
  dx   .+= ou.σ * randn(Float32, length(ou.X)) |> gpu
  ou.X .+= dx
end

# Noise scale
τ_ = 25
ϵ  = exp(-1f0 / τ_)
noise_scale = 1f0 / ACTION_BOUND

# ----------------------------- Model Architecture -----------------------------

w_init(dims...) = 6f-3rand(Float32, dims...) .- 3f-3

actor = Chain(Dense(STATE_SIZE, 400, relu),
	      Dense(400, 300, relu),
              Dense(300, ACTION_SIZE, tanh, initW=w_init),
              x -> x * ACTION_BOUND) |> gpu
actor_target = deepcopy(actor)

struct crit
  state_crit
  act_crit
  sa_crit
end

Flux.@treelike crit

function (c::crit)(state, action)
  s = c.state_crit(state)
  a = c.act_crit(action)
  c.sa_crit(relu.(s .+ a))
end

Base.deepcopy(c::crit) = crit(deepcopy(c.state_crit),
                              deepcopy(c.act_crit),
			      deepcopy(c.sa_crit))

critic = crit(Chain(Dense(STATE_SIZE, 400, relu), Dense(400, 300)) |> gpu,
              Dense(ACTION_SIZE, 300) |> gpu,
	      Dense(300, 1, initW=w_init) |> gpu)
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

function L2_loss(model)
  l2_loss = 0
  for p in params(model)
    l2_loss += sum(p .^ 2)
  end

  return L2_DECAY * l2_loss
end

opt_crit = ADAM(η_crit)
opt_act  = ADAM(η_act)

function replay()
  s, a, r, s′, s_mask = getData()
  
  # Update Critic
  a′ = data(actor_target(s′))
  v′ = data(critic_target(s′, a′))
  y = r .+ γ * v′ .* s_mask	# set v′ to 0 where s_ is terminal state

  v = critic(s, a)
  loss_crit = Flux.mse(y, v) + L2_loss(critic)
  
  zero_grad!(critic)
  Flux.back!(loss_crit)
  _update_params!(opt_crit, params(critic))
  
  # Update Actor
  actions = actor(s)
  crit_out = critic(s, actions)
  
  zero_grad!(actor)
  Flux.back!(-sum(crit_out))
  _update_params!(opt_act, params(actor))

  # Update Target models
  update_target!(actor_target, actor; τ = τ)
  update_target!(critic_target, critic; τ = τ)
end

# --------------------------- Helper Functions --------------------------------

# Stores tuple of state, action, reward, next_state, and done
remember(state, action, reward, next_state, done) =
  push!(memory, [data.((state, action, reward[1], next_state))..., done])

# Choose action according to policy PendulumPolicy
function action(state, train=true)
  state = reshape(data(state), size(state)..., 1)
  act_pred = data(actor(state |> gpu))
  if train
    act_pred .+= noise_scale * sample_noise(ou)
  end
  clamp.(act_pred, -ACTION_BOUND, ACTION_BOUND) # returns action
end

function episode!(env, train=true)
  total_reward = 0f0
  s = reset!(env)
  for ep=1:MAX_EP_LENGTH
    a = action(s, train)
    s′, r, done, _ = step!(env, a)
    total_reward += data(r)[1]
    s = s′ 
    if train    
      remember(s, a, r, s′, done)
      replay()
    end
  end
  total_reward
end

# -------------------------------- Testing -------------------------------------

# Returns average score over 100 episodes
function test()
  score_mean = 0f0
  for _=1:100
    total_reward = episode!(env, false)
    score_mean += total_reward / 100
  end
  return score_mean
end

# ------------------------------ Training --------------------------------------

# Populate memory with random actions

s = reset!(env)
for e=1:MIN_EXP_SIZE
  global s
  a = 2rand(Float32) * ACTION_BOUND - ACTION_BOUND
  s′, r, done, _ = step!(env, a)
  remember(s, a, r, s′, done)
  s = s′
end

for e=1:MAX_EP
  global noise_scale
  total_reward = episode!(env)
  total_reward = @sprintf "%9.3f" total_reward
  print("Episode: $e | Score: $total_reward | ")
  score_mean = test()
  score_mean = @sprintf "%9.3f" score_mean
  println("Mean score over 100 test episodes: $score_mean")
  noise_scale *= ϵ
end
