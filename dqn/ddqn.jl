using Reinforce:CartPole, actions, reset!, Episode, finished
import Reinforce.action
using Flux, StatsBase, Plots

gr()

#Define custom policy for choosing action
mutable struct CartPolePolicy <: Reinforce.AbstractPolicy end

#Load game environment
env = CartPole()

#Parameters
EPISODES = 3000
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

huber_loss(x, y) = mean(sqrt.(1+(x-y).^2)-1)

#Model Architecture

model = Chain(Dense(STATE_SIZE, 24, σ), Dense(24, 24, σ), Dense(24, ACTION_SIZE))
loss(x, y) = huber_loss(model(x), y)
opt = ADAM(params(model), η)
fit_model(dataset) = Flux.train!(loss, dataset, opt)

#Target model Architecture
target_model = Chain(Dense(STATE_SIZE, 24, σ), Dense(24, 24, σ), Dense(24, ACTION_SIZE))

function remember(state, action, reward, next_state, done)
    if length(memory) == MEM_SIZE
        deleteat!(memory, 1)
    end
    push!(memory, (state, action, reward, next_state, done))
end

function update_target()
    for i in eachindex(params(target_model))
        for j in eachindex(params(target_model)[i].data)
            params(target_model)[i].data[j] = params(model)[i].data[j]
        end
    end
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
        target = model(state).data

        if done
            target[action, 1] = reward
        else
            a = model(next_state)[:, 1]
            t = target_model(next_state)[:, 1]
            target[action, 1] = reward + γ * t.data[Flux.argmax(a)]
        end

        dataset = zip(state, target)
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
        reward = !done ? reward : -1 #Penalty of -10 if game is over
        remember(state, action, reward, next_state, done)
    end
    ep.total_reward
end

update_target()

for e=1:EPISODES
    reset!(env)
    total_reward = episode!(env, CartPolePolicy())
    update_target()
    println("Episode: $e/$EPISODES | Score: $total_reward | ϵ: $ϵ")
    if length(memory) >= BATCH_SIZE
        replay()
    end
end
