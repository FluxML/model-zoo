using Flux, Statistics, Trebuchet
using Flux.Tracker
using Flux.Optimise: update!

# using CuArrays

using BSON: @save, @load

using Distributions: Normal, sample
using DataStructures: CircularBuffer
using Printf

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

# --------------------------------- Parameters ------------------------------- #

STATE_SIZE = 3
ACTION_SIZE = 2
BATCH_SIZE = 128
MEM_SIZE = 1000000
MAX_EP = 100000
MAX_DIST  = 500	# Maximum target distance
MAX_SPEED =  10 # Maximum wind speed

# ----------------------------- Hyperparameters ------------------------------ #

γ = 99f-2     	# Discount rate
τ = 1f-2 		# For running average while updating target networks
η_act = 1f-4    # Learning rate of actor
η_crit = 1f-3	# Learning rate of critic
noise = Normal(0, 1)
noise_scale = 1f-1

# ----------------------------- Model Architecture --------------------------- #

actor = Chain(Dense(STATE_SIZE, 256, relu), Dense(256, 512, relu),
	      Dense(512, 256, relu), Dense(256, ACTION_SIZE, relu)) |> gpu
actor_target =  deepcopy(actor)

critic = Chain(Dense(STATE_SIZE + ACTION_SIZE, 256, relu), Dense(256, 512, relu),
	       Dense(512, 256, relu), Dense(256, 1)) |> gpu
critic_target = deepcopy(critic)

# ---------------------------- Param Update Functions ------------------------ #

function nullify_grad!(p)
  if typeof(p) <: TrackedArray
    p.grad .= 0.0f0
  end
  return p
end

function zero_grad!(model)
  model = mapleaves(nullify_grad!, model)
end

function update_target!(target, model; τ = 1.0f0)
  for (p_t, p_m) in zip(params(target), params(model))
    p_t.data .= (1.0f0 - τ) * p_t.data .+ τ * p_m.data
  end
end

# -------------------------------- DDPG -------------------------------------- #

opt_crit = ADAM(η_crit)
opt_act = ADAM(η_act)

function trainer()
  # Getting data in shape
  minibatch = sample(memory, BATCH_SIZE)
  x = hcat(minibatch...)

  s = hcat(x[1, :]...) |> gpu
  a = hcat(x[2, :]...) |> gpu
  s′ = hcat(x[3, :]...) |> gpu
  r = hcat(x[4, :]...) |> gpu
  s_mask = .!hcat(x[5, :]...) |> gpu

  a′ = actor_target(s′).data
  crit_tgt_in = vcat(s′, a′)
  v′ = critic_target(crit_tgt_in).data
  y = r .+ γ * v′ .* s_mask	# set v′ to 0 where s_ is terminal state

  crit_in = vcat(s, a)
  v = critic(crit_in)
  loss_crit = Flux.mse(y, v)

  # Update Actor
  actions = actor(s)
  crit_in = param(vcat(s, actions.data))
  crit_out = critic(crit_in)

  Flux.back!(sum(crit_out))

  act_grads = -crit_in.grad[4:5, :]
  zero_grad!(actor)
  Flux.back!(actions, act_grads)  # Chain rule
  update!(opt_act, params(actor))

  # Update Critic
  zero_grad!(critic)
  Flux.back!(loss_crit)
  update!(opt_crit, params(critic))

end

# --------------------------------- Env Functions ---------------------------- #

memory = CircularBuffer{Any}(MEM_SIZE)

# stores the tuple of state, action, reward, next_state, and done
remember(state, action, next_state, reward, done) =
	push!(memory, [state, action, next_state, reward, done])

function action(state, train=true)
  state = reshape(state, size(state)..., 1)
  act_pred = cpu(actor(state |> gpu).data) .+
  					train * noise_scale * Float32.(rand(noise, ACTION_SIZE))
  act_pred  # returns action
end

function reward(target_dist, threshold, wind_speed, release_angle, weight)
  ws, ra, w = Float64.([wind_speed, release_angle, weight])
  t = TrebuchetState(wind_speed=ws, release_angle=ra, weight=w)
  simulate(t)
  actual_dist = Float32(t.sol.Projectile[end][1])
  r = (target_dist - actual_dist) / target_dist
  return max(-1f0, threshold - abs(r))
end

function episode(train=true)
  total_reward = 0
  threshold = 1f0

  wind_speed  = (2rand(Float32) - 1) * MAX_SPEED
  target_dist = (2rand(Float32) - 1) * MAX_DIST
  s = [wind_speed, target_dist, 100threshold]

  attempts = 0

  while threshold > 0
    a = action(s, train)

    r = reward(target_dist, threshold, wind_speed, a[1], a[2])
    total_reward += r * (r > 0)

    threshold -= 1f-2
    s′ = r < 0 ? [wind_speed, target_dist, 100threshold] : zeros(Float32, 3)
    train && remember(s, a, s′, r, r < 0)

    s .= s′
    attempts += 1

    if length(memory) >= 10000 && train
      trainer()
      update_target!(actor_target, actor; τ=τ)
      update_target!(critic_target, critic; τ=τ)
    end

    r < 0 && break
  end

  total_reward, attempts
end

# ------------------------------ Saving & Loading ---------------------------- #
#=
function saveModel()
  act_wts = cpu.(Tracker.data.(params(actor)))
  @save "models/actor.bson" act_wts

  crit_wts = cpu.(Tracker.data.(params(critic)))
  @save "models/critic.bson" crit_wts
  println("Model saved")
end

function loadModel()
  @load "models/actor.bson" act_wts
  @load "models/critic.bson" crit_wts

  Flux.loadparams!(actor, gpu.(act_wts))
  Flux.loadparams!(critic, gpu.(crit_wts))
  println("Model loaded")
end
=#
# ------------------------------- Training ----------------------------------- #

scores   = CircularBuffer{Float32}(1000)
attempts = CircularBuffer{Int}(1000)

println("Training starts...")
for e = 1:MAX_EP
  total_reward, att = episode()
  push!(scores, total_reward)
  push!(attempts, att)
  avg_score, avg_attempts = mean.([scores, attempts])

  print("Episode: $(@sprintf("%6d", e)) || ")
  print("Score: $(@sprintf("%8.5f", total_reward)) | ")
  print("Avg score: $(@sprintf("%7.5f", avg_score)) || ")
  print("Attempts: $(@sprintf("%2d", att)) | ")
  println("Avg attempts: $(@sprintf("%7.5f", avg_attempts)) || ")

  #saveModel()
end

println("Training ends...")

#------------------------------------- Testing ------------------------------- #
empty!(scores)
empty!(attempts)

println("Testing starts...")

for e = 1:1000
  total_reward, att = episode(false)
  push!(scores, total_reward)
  push!(attempts, att)
  avg_score, avg_attempts = mean.([scores, attempts])

  print("Episode: $(@sprintf("%6d", e)) || ")
  print("Score: $(@sprintf("%8.5f", total_reward)) | ")
  print("Avg score: $(@sprintf("%7.5f", avg_score)) || ")
  print("Attempts: $(@sprintf("%2d", att)) | ")
  println("Avg attempts: $(@sprintf("%7.5f", avg_attempts))")
end
