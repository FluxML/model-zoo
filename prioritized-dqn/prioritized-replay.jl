include("memory.jl")

using Reinforce: actions, reset!, Episode, finished
using Reinforce.CartPole
import Reinforce.action
using Flux, StatsBase, Plots
using Flux: params

gr()

#Define custom policy for choosing action
mutable struct CartPolePolicy <: Reinforce.AbstractPolicy end

#Load game environment
env = CartPole()

#Parameters
EPISODES = 10000
STATE_SIZE = length(env.state)
ACTION_SIZE = length(actions(env, env.state))
MEM_SIZE = 5000
BATCH_SIZE = 32
γ = 0.95    # discount rate
ϵ = 0.  # exploration rate
ϵ_max = 0.95
ϵ_incr = 0.00005
η = 0.005   #learning rate

replace_target_iter = 500
learn_step_counter = 0
steps = 0

memory = Memory(MEM_SIZE, STATE_SIZE)

#Model Architecture
model = Chain(Dense(STATE_SIZE, 24, σ), Dense(24, 24, σ), Dense(24, ACTION_SIZE))
target_model = Chain(Dense(STATE_SIZE, 24, σ), Dense(24, 24, σ), Dense(24, ACTION_SIZE))

#Loss
function loss(x, y, ISWeights)
    sq_diff = (x - y) .^ 2
    cost = mean(sq_diff * ISWeights')
    return cost
end

#Absolute Error (Used for SumTree)
abs_errors(x, y) = sum(abs.(x - y), 1)

#Optimizer
opt = Flux.RMSProp(params(model), η)

function replace_target_op()
    target_model = deepcopy(model)
end

function remember(state, action, reward, next_state, done)
    transition = vcat(state[:, 1], [action, reward], next_state[:, 1])
    store!(memory, transition)
end

function action(policy::CartPolePolicy, reward, state, action)
    if rand() <= ϵ
        act_values = model(state)
        return Flux.argmax(act_values)[1]  # returns action
    end
    return rand(1:ACTION_SIZE)
end

#Render the environment
on_step(env::CartPole) = gui(plot(env))

function learn()
    global learn_step_counter, ϵ, ϵ_max, ϵ_incr
    if learn_step_counter % replace_target_iter == 0
        replace_target_op()
    end

    tree_idx, batch_memory, ISWeights = mem_sample(memory, BATCH_SIZE)

    s_ = batch_memory[end - STATE_SIZE + 1:end, :]
    s = batch_memory[1:STATE_SIZE, :]

    q_next, q_curr = target_model(s_), model(s)

    q_target = q_curr.data
    eval_act_index = Int32.(batch_memory[STATE_SIZE + 1, :])
    reward = batch_memory[STATE_SIZE + 2, :]

    for i = 1:BATCH_SIZE
        q_target[eval_act_index[i], i] = reward[i] + γ * maximum(q_next[:, i])
    end

    cost = loss(q_curr, q_target, ISWeights)
    Flux.back!(cost)
    opt()

    abs_error = abs_errors(q_curr, q_target).data
    batch_update!(memory, tree_idx, abs_error)     # update priority
    ϵ +=  ϵ < ϵ_max ? ϵ_incr : 0.
    learn_step_counter += 1
end

function episode!(env, policy = RandomPolicy(); stepfunc = on_step, kw...)
    ep = Episode(env, policy)
    global steps

    for (s, a, r, s_) in ep
        on_step(ep.env)
        state, action, reward, next_state = s, a, r, s_

        state = reshape(state, STATE_SIZE, 1)

        next_state = reshape(next_state, STATE_SIZE, 1)

        done = finished(ep.env, next_state) #check if game is over
        reward = done ? -10 : reward
        remember(state, action, reward, next_state, done)
        steps += 1
        #if steps > MEM_SIZE
        learn()
        #end
    end
    ep.total_reward
end

for e=1:EPISODES
    reset!(env)
    total_reward = episode!(env, CartPolePolicy())
    println("Episode: $e/$EPISODES | Score: $total_reward | ϵ: $ϵ")
end
