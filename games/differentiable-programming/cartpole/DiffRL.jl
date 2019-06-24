using Flux, Gym, Printf, Zygote
using Zygote: @adjoint
using Flux.Optimise: update!
using Statistics: mean
#using CuArrays

import Base.sign

#Load game environment

env = make("CartPole-v0")
reset!(env)

#ctx = Ctx(env)

#display(ctx.s)
#using Blink# when not on Juno
#body!(Blink.Window(), ctx.s)

# ----------------------------- Parameters -------------------------------------

STATE_SIZE = length(env._env.state)
ACTION_SIZE = length(env._env.action_space)
MAX_TRAIN_REWARD = env._env.x_threshold * env._env.θ_threshold_radians
SEQ_LEN = 8

# Optimiser params
η = 3f-2
# ------------------------------ Model Architecture ----------------------------
sign(x) = Base.sign.(x)
@adjoint sign(x) = sign(x), x̄ -> (x̄,)

model = Chain(Dense(STATE_SIZE, 24, relu),
              Dense(24, 48, relu),
              Dense(48, 1, tanh), x->sign(x)) |> gpu

opt = ADAM(η)

action(state) = state |> model |> (model_output) -> (3 .+ model_output) / 2

loss(rewards) = Flux.mse(rewards, MAX_TRAIN_REWARD)

# ----------------------------- Helper Functions -------------------------------

function train_reward(env::EnvWrapper)
    s = env._env.state
    x, ẋ, θ, θ̇  = s
    # Custom reward for training
    # Product of Triangular function over x-axis and θ-axis
    # Min reward = 0, Max reward = env.x_threshold * env.θ_threshold_radians
    x_upper = env._env.x_threshold - x
    x_lower = env._env.x_threshold + x

    r_x     = max(0f0, min(x_upper, x_lower))

    θ_upper = env._env.θ_threshold_radians - θ
    θ_lower = env._env.θ_threshold_radians + θ

    r_θ     = max(0f0, min(θ_upper, θ_lower))

    return r_x * r_θ
end

function μEpisode(env::EnvWrapper)
    l = 0
    for frames ∈ 1:SEQ_LEN
        #render(env, ctx)
        #sleep(0.01)
        a = action(env._env.state)
        s′, r, done, _ = step!(env, a)

        if trainable(env)
            l += loss(train_reward(env))
        end
        game_over(env) && break
    end
    return l
end

function episode!(env::EnvWrapper)
    reset!(env)
    while !game_over(env)
        if trainable(env)
            grads = gradient(()->μEpisode(env), params(model))
            update!(opt, params(model), grads)
        else
            μEpisode(env)
        end
    end
    env.total_reward
end

# -------------------------------- Testing -------------------------------------

function test(env::EnvWrapper)
    score_mean = 0f0
    testmode!(env)
    for _=1:100
        total_reward = episode!(env)
        score_mean += total_reward / 100
    end
    testmode!(env, false)
    return score_mean
end

# ------------------------------ Training --------------------------------------

e = 1
while true
    global e
    total_reward = @sprintf "%6.2f" episode!(env)
    print("Episode: $e | Score: $total_reward | ")

    score_mean = test(env)
    score_mean_str = @sprintf "%6.2f" score_mean
    print("Mean score over 100 test episodes: " * score_mean_str)

    println()

    if score_mean > env.reward_threshold
        println("CartPole-v0 solved!")
    break
    end
    e += 1
end
