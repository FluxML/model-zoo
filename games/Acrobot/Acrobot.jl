import LearnBase.update!
using OpenAIGym
using Distributions
using Flux, Reinforce

#--------------------------------SumTree Implementation-------------------------------#
mutable struct SumTree
    capacity::Int
    tree::Array{Float32, 1}
    data::Array{Any, 2}
    write::Int

    function SumTree(cap::Int, s_size::Int)
        capacity = cap
        tree = zeros(2 * capacity - 1)
        data = zeros(2 * s_size + 2, capacity)   #Storing state, action, reward, next_state
        write = 0
        new(capacity, tree, data, write)
    end
end

function add!(sumt::SumTree, p, data)
    ind = sumt.write + sumt.capacity - 1
    sumt.data[:, sumt.write + 1] = data

    update!(sumt, ind, p)

    sumt.write += 1
    if sumt.write >= sumt.capacity
        sumt.write = 0
    end
end

function update!(sumt::SumTree, ind::Int, p)
    change = p - sumt.tree[ind + 1]
    sumt.tree[ind + 1] = p

    while ind != 0
        ind = div(ind - 1, 2)
        sumt.tree[ind + 1] += change
    end
end

function leaf(sumt::SumTree, v)
    parent_ind = 0
    leaf_ind = 0
    while true
      lchild_ind = 2 * parent_ind + 1        
      rchild_ind = lchild_ind + 1
      if lchild_ind >= length(sumt.tree)        
        leaf_ind = parent_ind
        break
      elseif v <= sumt.tree[lchild_ind + 1]
        parent_ind = lchild_ind
      else
        v -= sumt.tree[lchild_ind + 1]
        parent_ind = rchild_ind
      end
    end
  
    data_ind = leaf_ind - sumt.capacity + 1
    return leaf_ind, sumt.tree[leaf_ind + 1], sumt.data[:, data_ind + 1]
end

function total(sumt::SumTree)
    return sumt.tree[1]
end

#---------------------------------------------Memory Implementation------------------------------------#
mutable struct Memory
    ϵ::Float32
    α::Float32
    β::Float32
    β_increment_per_sampling::Float32
    abs_error_upper::Float32
    sumt::SumTree

    function Memory(cap::Int, s_size::Int)
        ϵ = 0.01
        α = 0.6
        β = 0.4
        β_increment_per_sampling = 0.001
        abs_error_upper = 1.0
        sumt = SumTree(cap, s_size)
        new(ϵ, α, β, β_increment_per_sampling, abs_error_upper, sumt)
    end
end

function store!(mem::Memory, transition)
    max_p = maximum(mem.sumt.tree[end - mem.sumt.capacity + 1:end])
    max_p = max_p == 0 ? mem.abs_error_upper : max_p
    add!(mem.sumt, max_p, transition)
end

function mem_sample(mem::Memory, n)
    b_idx = Array{Int, 1}(undef, n)
    b_memory = Array{Float32, 2}(undef, length(mem.sumt.data[:, 1]), n)
    ISWeights = Array{Float32, 1}(undef, n)

    priority_seg = total(mem.sumt) / n
    mem.β = min(1., mem.β + mem.β_increment_per_sampling)
    min_prob = minimum(mem.sumt.tree[end - mem.sumt.capacity + 1:end]) / total(mem.sumt)

    for i = 0:n-1
        a,b = priority_seg * i, priority_seg * (i + 1)
        v = rand(Uniform(a, b))
        idx, p, data = leaf(mem.sumt, v)
        prob = p / total(mem.sumt)
        ISWeights[i+1] = float(min_prob/prob) ^ mem.β
        b_idx[i + 1], b_memory[:, i + 1] = idx, data
    end

    return b_idx, b_memory, ISWeights
end

function batch_update!(mem::Memory, tree_idx, abs_errors)
    abs_errors .+= mem.ϵ  # convert to abs and avoid 0
    clipped_errors = min.(abs_errors, mem.abs_error_upper)
    ps = clipped_errors .^ mem.α
    for (ti, p) in zip(tree_idx, ps)
      ti = convert(Int64, ti)
      update!(mem.sumt, ti , p)
    end
end

#--------------------------Initialize Environment------------------------------#
env = GymEnv("Acrobot-v1")

#---------------------------------------Parameters-------------------------------------#
STATE_SIZE = length(env.state)
ACTION_SIZE = length(env.actions)
MEM_SIZE = 10000
EPISODES = 10000
BATCH_SIZE = 32
FREQ = 250

γ = 0.99
η = 0.0003

ϵ = 1.0
ϵ_min = 0.01
ϵ_decay = 0.995

C = 0
memory = Memory(MEM_SIZE, STATE_SIZE)
frames = 0

#---------------------------------Model Architecture------------------------------------#
model = Chain(Dense(STATE_SIZE, 24), Dense(24, 48), Dense(48, ACTION_SIZE))
model_target = deepcopy(model)

#-----------------------Loss Function--------------------#
function loss(x,y, ISWeights)
    x, y, ISWeights
    squared_diff = (x .- y) .^ 2
    reshaped_ISweights = reshape(ISWeights, 1, length(ISWeights))
    cost = mean(squared_diff .* reshaped_ISweights)
    return cost
end

abs_errors(x, y) = sum(abs.(x - y), dims=1)

opt() = ADAM(η)

function remember(state, action, reward, state_next, done)
    global frames
    transit = vcat(state, [action, reward], state_next)
    store!(memory, transit)
    frames += 1
end

function act(state)
    if rand() <= ϵ
        return rand(1:ACTION_SIZE) - 1
    end
    act_vals = model(state)
    return argmax(act_vals) - 1
end

#-----------------------------Prioritized Replay---------------------------#
function prioritized_replay()
    global C, ϵ, model_target

    if C == 0
        model_target = deepcopy(model)
    end

    tree_ind, batch_memory, ISWeights = mem_sample(memory, BATCH_SIZE)

    states = batch_memory[1:STATE_SIZE, :]
    next_states = batch_memory[end - STATE_SIZE + 1:end, :]

    q_next, q_curr = model_target(next_states), model(states)
    q_target = q_curr.data
    eval_act_index = Int32.(batch_memory[STATE_SIZE + 1, :])
    reward = batch_memory[STATE_SIZE + 2, :]

    for i = 1:BATCH_SIZE
        q_target[eval_act_index[i], i] = reward[i] + γ * maximum(q_next[:, i].data)
    end

    cost = loss(q_curr, q_target, ISWeights)
    Flux.back!(cost)
    opt()
  
    C = (C + 1) % FREQ
  
    abs_error = abs_errors(q_curr, q_target).data
    batch_update!(memory, tree_ind, abs_error)
  
    ϵ *= ϵ > ϵ_min ? ϵ_decay : 1.0
end

#--------------------------------------Testing-----------------------------------#
function test(epis=125)
    for ep=1:epis
        reset!(env)
        state = convert(Array, env.state)
        total_reward = 0
        while true
            #OpenAIGym.render(env)
            action = act(state)
            reward, next_state = step!(env, state, action)
            reward = !env.done ? reward : -reward
            total_reward += reward
            next_state = convert(Array, next_state)
            state = next_state
            if env.done
                println("Episode: $ep | Score: $total_reward | ϵ: $ϵ")
                break
            end
        end
    end
end

#----------------------------------Training----------------------------------#
for ep=1:EPISODES
    reset!(env)
    state = convert(Array, env.state)
    total_reward = 0
    while true
        OpenAIGym.render(env)
        action = act(state)
        reward, next_state = step!(env, state, action)
        reward = !env.done ? reward : -reward
        total_reward += reward
        next_state = convert(Array, next_state)
        remember(state, action + 1, reward, next_state, env.done)
        state = next_state
        if env.done
            println("Episode: $ep | Score: $total_reward | ϵ: $ϵ")
            break
        end
    end
    if frames >= MEM_SIZE
        prioritized_replay()
    end
end

test(150)