using Flux, Statistics, Trebuchet, Printf
using Flux.Tracker: data, grad, gradient
using Flux.Optimise: _update_params!
using DataStructures: CircularBuffer
using Distributions: sample

#using CuArrays

#=
# Description of the problem:
    There is Trebuchet, which throws a mass to a target. The mass is to be
    released at an angle, and at certain velocity so that it lands on the target.
    The velocity of release is determined by the counterweight of the Trebuchet.
    Given conditions of environment we are required to predict the angle of
    release and counterweight.

    The problem is gamified by introducing a threshold. The player gets 99
    attempts. The first attempt has threshold value of 1. Threshold determines
    the tolerable relative error between actual distance travelled by the Projectile
    and the target distance. This relative error has to be less than the threshold
    in order to continue playing the game. The game gets tougher at each attempts
    by reduction of threshold by 0.01.

# Input:  Wind speed,   Target distance
# Output: ReleaseAngle, Weight
=#

# ----------------------------- Parameters -------------------------------------

STATE_SIZE = 2
ACTION_SIZE = 2

DIST  = (2f1, 1f2)	# Maximum target distance
SPEED =   5f0 # Maximum wind speed

ACTION_BOUND = [DIST[2]-DIST[1], SPEED]
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
noise_scale = 1f0

# ----------------------------- Model Architecture -----------------------------

w_init(dims...) = 6f-3rand(Float32, dims...) .- 3f-3

actor = Chain(Dense(STATE_SIZE, 400, relu),
	      	  Dense(400, 300, relu),
              Dense(300, ACTION_SIZE, tanh, initW=w_init)) |> gpu
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
  @show size(s)
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

lerp(x, lo, hi) = x*(hi-lo)+lo

target() = [randn() * SPEED, lerp(rand(), DIST...)]

# Stores tuple of state, action, reward, next_state, and done
remember(state, action, reward, next_state, done) =
  push!(memory, [data.((state, action, reward[1], next_state))..., done])

function action(state, train=true)
  state = reshape(state, size(state)..., 1)
  act_pred = data(actor(state |> gpu)) |> cpu
  if train
    act_pred .+= noise_scale * sample_noise(ou)
  end
  angle = clamp.(act_pred[1:1, :], -π, π) # returns action

  return vcat(angle, act_pred[2:2, :])
end

function shoot(wind, angle, weight)
  Trebuchet.shoot((wind, Trebuchet.deg2rad(angle), weight))[2]
end

function reward(wind, target_dist, angle, weight)
  actual_dist = shoot(wind, angle, weight)
  -(target_dist - actual_dist) ^ 2
end

function step!(state, action)
  r = reward(state..., action...)
  return r
end

function episode(train=true)
  wind, target_dist = target()
  s = [wind_speed, target_dist]
  a = action(s, train)
  r = step!(s, a)

  if train
    remember(s, a, zeros(Float32, 2), r, true)
    replay()
  end

  return √-r
end
# -------------------------------- Testing -------------------------------------

# Returns average score over 100 episodes
function test()
  score_mean = 0f0
  for _=1:100
    total_reward = episode(false)
    score_mean += total_reward / 100
  end
  return score_mean
end

# ------------------------------ Training --------------------------------------

# Populate memory with random actions

for e=1:MIN_EXP_SIZE
  s = target()
  a = 2rand(Float32, ACTION_SIZE) .* ACTION_BOUND .- ACTION_BOUND
  r = step!(s, a)
  remember(s, a, r, zeros(Float32, 2), true)
end

for e=1:MAX_EP
  global noise_scale
  total_reward = episode(true)
  total_reward = @sprintf "%9.3f" total_reward
  print("Episode: $e | Score: $total_reward | ")
  score_mean = test()
  score_mean = @sprintf "%9.3f" score_mean
  println("Mean score over 100 test episodes: $score_mean")
  noise_scale *= ϵ
end
