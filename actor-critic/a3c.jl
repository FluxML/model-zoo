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

γ = 0.99

cᵥ = 0.5			# v loss coefficient
cₑ = 0.01 # entropy coefficient

ϵ_g = 0.99
ϵ_min = 0.01
ϵ_decay = 0.995

mutable struct worker
  env::Reinforce.CartPole
  S::Array{Array{Float64,1},1}  #states
  A::Array{Int,1} #actions
  R::Array{Int, 1}  #Rewards
  S′::Array{Array{Float64,1},1} #rewards
  state_masks::Array{Bool, 1} #false if state is terminal, else true

  ϵ_START::Float64  #ϵ is exploration rate
  ϵ_STOP::Float64
  ϵ_STEPS::Int
  t::Int

  function worker(ϵ_START = 0.4, ϵ_STOP = 0.15, ϵ_STEPS::Int = 75000)
    new(CartPole(), Array{Array{Float64,1},1}(), Array{Int,1}(),
    Array{Int, 1}(), Array{Array{Float64,1},1}(), Array{Bool, 1}(), ϵ_START,
    ϵ_STOP, ϵ_STEPS, 0)
  end
end

# ----------------------------- Model Architecture -----------------------------

base = Chain(Dense(STATE_SIZE, 24, relu), Dense(24, 24, relu))
value = Dense(24, 1)
policy = Dense(24, ACTION_SIZE)

# ----------------------------- Loss ------------------------------------------

# Policy Loss
function loss_π(π, v, action, rₜ)
    logπ = log.(mean(π .* action, 1) + 1e-10)
    advantage = rₜ .- v
    -logπ .* advantage.data #to stop backpropagation through advantage
end

# Value loss
lossᵥ(v, rₜ) = cᵥ * (rₜ .- v) .^ 2

# Entropy loss
lossₑ(π) = cₑ * mean(π .* log.(π + 1e-10), 1)

# Total Loss = Policy loss + Value Loss + Entropy
function loss(s, a, r, s′, s_mask)
  base_out = base(s)
  v = value(base_out)
  π = softmax(policy(base_out))

  v′ = value(base(s′))


  rₜ = r .+ γ * v′ .* s_mask	# set v to 0 where s_ is terminal state

  mean(loss_π(π, v, a, rₜ) .+ lossᵥ(v, rₜ) .+ lossₑ(π))
end

# --------------------------- Training ----------------------------------------

opt = RMSProp(params(base) ∪ params(value) ∪ params(policy), η)

function train(w::worker)
	states = hcat(w.S...)
  next_states = hcat(w.S′...)
  actions = onehotbatch(w.A, 1:ACTION_SIZE)

  back!(loss(states, actions, w.R, next_states, w.state_masks))
end

# --------------------------- Helper Functions --------------------------------

function get_ϵ(w::worker)
  if w.t >= w.ϵ_STEPS
    return w.ϵ_STOP
  else
    return w.ϵ_START + w.t * (w.ϵ_STOP - w.ϵ_START) / w.ϵ_STEPS	# linearly interpolate
  end
end

# stores the tuple of state, action, reward, next_state, and done
function remember(w::worker, s, a, r, s′, done)
  push!(w.S, s)
  push!(w.A, a)
  push!(w.R, r)
  push!(w.S′, s′)
  push!(w.state_masks, !done)
end

# Choose action according to policy CartPolePolicy
function action(p::CartPolePolicy, reward, state, action)
  global ϵ_g, ϵ_decay
  if rand() < ϵ_g
      return rand(1:ACTION_SIZE)
  end

  if ϵ_g > ϵ_min
    ϵ_g *= ϵ_decay
  end

  π = policy(base(state))
  sample(1:ACTION_SIZE, Weights(π.data)) # returns action
end


#Render the environment
on_step(env::CartPole, niter, sars) = gui(plot(env))

function episode!(w::worker, p = RandomPolicy(); stepfunc = on_step, kw...)
    ep = Episode(w.env, p) # Runs an episode with policy p

    for sars in ep
        stepfunc(ep.env, ep.niter, sars)
        s, a, r, s′ = sars
        done = finished(ep.env, s′) #check if game is over

        if done
          r = -10 #Penalty of -10 if game is over
          s′ = zeros(Float64, 4)
        end

        remember(w, s, a, r, s′, done)
        w.t += w.t == 75000 ? 0 : 1
    end

    ep.total_reward
end

e = 1

work = [worker(), worker(), worker()

function empty!(w::worker)
  empty!(w.S)
  empty!(w.A)
  empty!(w.R)
  empty!(w.S′)
  empty!(w.state_masks)
end

while true
  for (w_id, w) in enumerate(work)
    reset!(w.env)
    total_reward = episode!(w, CartPolePolicy())
    ϵ = get_ϵ(w)
    println("Worker: $w_id | Episode: $e | Score: $total_reward | ϵ: $ϵ ")
    train(w)
    opt()

    empty!(w)
    e += 1
  end
end
