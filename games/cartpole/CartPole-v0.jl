using DataStructures
import Reinforce
using Reinforce:CartPoleV0, actions, reset!, finished, step!
using Flux, StatsBase, Plots

gr()

#-------------Initialise game environment--------------#
env = CartPoleV0()

#---------------Parameters--------------#
EPISODES = 3000
STATE_SIZE = length(env.state)
ACTION_SIZE = length(actions(env, env.state))

BATCH_SIZE = 32

γ = 0.95                # discount rate
η = 0.001               # learning rate

ϵ = 1.0                 # exploration rate
ϵ_min = 0.01            # exploration minimum
ϵ_decay = 0.995         # exploration decay

memory = CircularBuffer{Any}(2000)             # used to remember past results

#-------------Model Architecture--------------#
model = Chain(Dense(STATE_SIZE, 24), Dense(24, 24), Dense(24, ACTION_SIZE))
loss(x, y) = Flux.mse(model(x), y)
opt = ADAM(η)
fit_model(dataset) = Flux.train!(loss, params(model), dataset, opt)

function remember(state, action, reward, next_state, done)
    push!(memory, (state, action, reward, next_state, done))
end

function act(state)
    if rand() <= ϵ
        return rand(1:ACTION_SIZE)
    end
    act_values = model(state).data
    return argmax(act_values)[1]  # returns action
end

function exp_replay()
    global ϵ
    minibatch = sample(memory, BATCH_SIZE, replace = false)

    for (state, action, reward, next_state, done) in minibatch
        target = reward
        if !done
            target += γ * maximum(model(next_state).data)
        end
        target_f = model(state).data
        target_f[action, 1] = target
        dataset = [(state, target_f)]
        fit_model(dataset)
    end

    if ϵ > ϵ_min
        ϵ *= ϵ_decay
    end
end

#------------------------Render the environment--------------------#
gui(plot(env))

#-----------------------Testing-----------------------#
function test(epis=125)
    for ep=1:epis
        reset!(env)
        state = env.state
        state = reshape(state, STATE_SIZE, 1) |> gpu
        step = 0
        while true
            step += 1
            gui(plot(env))
            action = act(state)
            reward, next_state = step!(env, state, action)
            done = finished(env,next_state)
            next_state = reshape(next_state, STATE_SIZE, 1)
            state = gpu(next_state)
            if done
                println("Episode: $ep/$epis | Score: $step | ϵ: $ϵ")
                break
            end
        end
    end
end

#------------------------Training----------------------------#
for e=1:EPISODES
    reset!(env)
    state = env.state
    state = reshape(state, STATE_SIZE, 1)
    step = 0
    while true
        step += 1
        gui(plot(env))
        action = act(state)
        reward, next_state = step!(env, state, action)
        done = finished(env, next_state) #check of game is over
        reward = !done ? reward : -reward #Penalty of -10 if game is over
        next_state = reshape(next_state, STATE_SIZE, 1)
        remember(state, action, reward, next_state, done)
        state = next_state
        if done
            println("Episode: $e/$EPISODES | Score: $step | ϵ: $ϵ")
            break
        end
    end
    if length(memory) >= BATCH_SIZE
        exp_replay()
    end
end

test()
