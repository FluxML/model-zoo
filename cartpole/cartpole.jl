using PyCall
using Flux
using StatsBase
@pyimport gym

#Load game environment
env = gym.make("CartPole-v1")

#Parameters
EPISODES = 1000
STATE_SIZE = env[:observation_space][:shape][1]
ACTION_SIZE = env[:action_space][:n]
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

function act(state)
    if rand() <= ϵ
        return rand(0:ACTION_SIZE-1)
    end
    act_values = model(state)
    return Flux.argmax(act_values)[1] - 1  # returns action
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
        target_f[action + 1, 1] = target
        dataset = zip(state, target_f)
        fit_model(dataset)
    end

    if ϵ > ϵ_min
        ϵ *= ϵ_decay
    end
end

env[:_max_episode_steps] = 10000 #A large number

for e=1:EPISODES
    state = env[:reset]()
    state = reshape(state, STATE_SIZE, 1)
    score = 0
    done = false
    while !done
        env[:render]()
        #Select action to perform
        action = act(state)
        #Perform action
        next_state, reward, done, _ = env[:step](action)
        reward = !done ? reward : -10 #Penalty of -10 if game is over
        next_state = reshape(next_state, STATE_SIZE, 1)
        remember(state, action, reward, next_state, done)
        state = next_state
        score += 1
    end
    println("Episode: $e/$EPISODES | Score: $score | ϵ: $ϵ")
    if length(memory) >= BATCH_SIZE
        replay()
    end
end
