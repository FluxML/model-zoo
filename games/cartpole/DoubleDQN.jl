using DataStructures
using BSON: @save, @load
import Reinforce
using Reinforce: CartPoleV0, actions, reset!, finished, step!
using Flux, CuArrays, StatsBase, Plots

gr()
ENV["GKSwstype"] = "100"

#---------------Initialize game environment----------------#
env = CartPoleV0()


#-------------------------Parameters-----------------------#
EPISODES = 400
STATE_SIZE = length(env.state)
ACTION_SIZE = length(actions(env, env.state))
REPLAY_MEMORY = 10000
MAX_STEPS = 300

BATCH_SIZE = 32

γ = 0.99                # discount rate
η = 0.0001              # learning rate

ϵ = 1.0                 # exploration rate
ϵ_min = 0.01            # exploration minimum
ϵ_decay = 0.99          # exploration decay

memory = CircularBuffer{Any}(REPLAY_MEMORY)


#-----------------------Model Architecture------------------------#
model = Chain(Dense(STATE_SIZE, 24, relu),
              Dense(24, 24, relu), 
              Dense(24, ACTION_SIZE)) |> gpu

loss(x, y) = Flux.mse(model(x), y)
opt = ADAM(η)

fit_model(dataset) = Flux.train!(loss, params(model), dataset, opt)

model_target = deepcopy(model)

"""Save sample (s, a, r, s′) to replay memory"""
function remember(state, action, reward, next_state, done)
    push!(memory, (state, action, reward, next_state, done))
end


"""Get action from model using epsilon-greedy policy"""
function act(state, ϵ)
    rand() <= ϵ && return rand(1:ACTION_SIZE)
    q_values = model(state |> gpu).data
    return argmax(q_values)
end

"""Sample from replay memory, train model, update exploration"""
function replay()
    length(memory) < BATCH_SIZE && return nothing
    
    batch_size = min(BATCH_SIZE, length(memory))
    minibatch = sample(memory, batch_size, replace=false)
    
    sb, ab, rb, s′b, db = collect.(zip(minibatch...))
    sb = hcat(sb...) |> gpu
    s′b = hcat(s′b...) |> gpu
    
    qb_target = model(sb).data
    
    actions = argmax(model(s′b).data)
    qb_learn = model_target(s′b).data
    qb_learned = getindex.(Ref(qb_learn), actions)
    
    qb_learned = ifelse.(db, rb, rb .+ γ .* cpu(qb_learned))
    setindex!.(Ref(qb_target), qb_learned, ab)
    
    dataset = [(sb, qb_target)]
    fit_model(dataset)
    
    GC.gc(); # CuArrays.clearpool()
    
    global ϵ
    ϵ = ϵ_min + (ϵ - ϵ_min)*ϵ_decay
end

#----------------------------Training & Testing---------------------------#
best_score = 0.0
test_every, TEST = Integer(EPISODES/10), 10

for e=1:EPISODES
    reset!(env)
    state = env.state
    score = 0
    
    envs = []
    for step=1:MAX_STEPS
        push!(envs, deepcopy(env))
        
        action = act(state, ϵ)
        reward, next_state = step!(env, state, action)
        done = finished(env, next_state)
        reward = !done ? reward : -1
        score += reward
        
        remember(state, action, reward, next_state, done)
        
        state = next_state
        done && break
    end
    
    stats = "Episode: $e/$EPISODES | Score: $score | ϵ: $ϵ"
    
    if best_score <= score
        best_score = score
        println(stats); flush(stdout)
        @save "models/notebook4/model-$e-$score.bson" model
        anim = @animate for env in envs
            plot(env)
        end
        mp4(anim, "models/notebook4/env-$e-$score.mp4", fps=20, show_msg=false)
    else
        print(stats); flush(stdout); print("\r")
    end
    
    replay()
    model_target = deepcopy(model)
    
    if e % test_every == 0
        score = 0
        for i=1:TEST
            reset!(env)    
            state = env.state
            
            for step=1:MAX_STEPS
                action = act(state, ϵ_min)
                reward, state = step!(env, state, action)
                done = finished(env, state)
                reward = !done ? reward : -1 
                score += reward

                done && break
            end
        end
        
        score /= TEST
        println("#-- Avg Test Score $(Integer(e/test_every)) : $score --#")
        score >= 200 && break
    end
end

println("Done!")