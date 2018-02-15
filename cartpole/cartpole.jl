using Reinforce:CartPole, actions, reset!, Episode, finished
import Reinforce.action
using Flux, StatsBase, Plots

gr()

#Define custom policy for choosing action
type CartPolePolicy <: Reinforce.AbstractPolicy end

#Load game environment
env = CartPole()

#Parameters
EPISODES = 1000
STATE_SIZE = length(env.state)
ACTION_SIZE = length(actions(env, env.state))
MEM_SIZE = 2000
BATCH_SIZE = 32
γ = 0.95    # discount rate
ϵ = 1.0  # exploration rate
ϵ_min = 0.01
ϵ_decay = 0.995
η = 0.001   #learning rate

memory = [] #used to remember past results

#Model Architecture
model = Chain(Dense(STATE_SIZE, 24, σ), Dense(24, 24, σ), Dense(24, ACTION_SIZE))
loss(x, y) = Flux.mse(model(x), y)
opt = ADAM(params(model), η)
fit_model(dataset) = Flux.train!(loss, dataset, opt)

function remember(state, action, reward, next_state, done)
    if length(memory) == MEM_SIZE
        deleteat!(memory, 1)
    end
    push!(memory, (state, action, reward, next_state, done))
end

function action(policy::CartPolePolicy, reward, state, action)
    if rand() <= ϵ
        return rand(1:ACTION_SIZE)
    end
    act_values = model(state)
    return Flux.argmax(act_values)[1]  # returns action
end

function replay()
    global ϵ
    minibatch = sample(memory, BATCH_SIZE, replace = false)

    for (state, action, reward, next_state, done) in minibatch
        target = reward
        if !done
            target += γ * maximum(model(next_state))
        end
        target_f = model(state).data
        target_f[action, 1] = target
        dataset = zip(state, target_f)
        fit_model(dataset)
    end

    if ϵ > ϵ_min
        ϵ *= ϵ_decay
    end
end

#Render the environment
on_step(env::CartPole, niter, sars) = gui(plot(env))

function episode!(env, policy = RandomPolicy(); stepfunc = on_step, kw...)
    ep = Episode(env, policy)
    for sars in ep
        stepfunc(ep.env, ep.niter, sars)
        state, action, reward, next_state = sars
        state = reshape(state, STATE_SIZE, 1)
        next_state = reshape(next_state, STATE_SIZE, 1)
        done = finished(ep.env, next_state) #check of game is over)
        reward = !done ? reward : -10 #Penalty of -10 if game is over
        remember(state, action, reward, next_state, done)
    end
    ep.total_reward
end

for e=1:EPISODES
    reset!(env)
    total_reward = episode!(env, CartPolePolicy())
    println("Episode: $e/$EPISODES | Score: $total_reward | ϵ: $ϵ")
    if length(memory) >= BATCH_SIZE
        replay()
    end
end
