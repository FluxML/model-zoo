import Base.show

mutable struct CartPoleEnv
    gravity
    masscart
    masspole
    total_mass
    length  # actually half the pole's length
    polemass_length
    force_mag
    τ   # seconds between state updates
    kinematics_integrator

    # Angle at which to fail the episode
    θ_threshold_radians
    x_threshold
    action_space
    observation_space
    viewer
    state

    steps_beyond_done
end

function CartPoleEnv()
    gravity = 98f-1
    masscart = 1f0
    masspole = 1f-1
    total_mass = masspole + masscart
    length = 5f-1 # actually half the pole's length
    polemass_length = (masspole * length)
    force_mag = 1f1
    τ = 2f-2  # seconds between state updates
    kinematics_integrator = "semi-implicit euler" # Or "euler"

    # Angle at which to fail the episode
    θ_threshold_radians = Float32(12 * 2 * π / 360)
    x_threshold = 24f-1

    # Angle limit set to 2θ_threshold_radians so failing observation is still within bounds
    high = [
        2x_threshold,
        maxintfloat(Float32),
        2θ_threshold_radians,
        maxintfloat(Float32)]

    action_space = 1:2
    observation_space = (-high, high)

    viewer = nothing
    state = nothing

    steps_beyond_done = nothing
    CartPoleEnv(
        gravity, masscart, masspole, total_mass, length, polemass_length,
        force_mag, τ, kinematics_integrator, θ_threshold_radians, x_threshold,
        action_space, observation_space, viewer, state, steps_beyond_done)
end

function step!(env::CartPoleEnv, action)
#    @assert action ∈ env.action_space "$action in ($(env.action_space)) invalid"
    state = env.state
    x, ẋ, θ, θ̇  = param.(state)
    #force = action == 2 ? env.force_mag : -env.force_mag
    action = param(action)
    force = action * env.force_mag  # action is +1 or -1
    cosθ = cos(θ)
    sinθ = sin(θ)
    temp = (force + env.polemass_length * θ̇  ^ 2 * sinθ) / env.total_mass
    θacc = (env.gravity*sinθ - cosθ*temp) /
           (env.length * (4f0/3 - env.masspole * cosθ ^ 2 / env.total_mass))
    xacc  = temp - env.polemass_length * θacc * cosθ / env.total_mass
    if env.kinematics_integrator == "euler"
        x_ = x + env.τ * ẋ
        ẋ_ = ẋ + env.τ * xacc
        θ_ = θ + env.τ * θ̇
        θ_dot_ = θ̇  + env.τ * θacc
    else # semi-implicit euler
        ẋ_ = ẋ + env.τ * xacc
        x_ = x + env.τ * ẋ_
        θ_dot_ = θ̇  + env.τ * θacc
        θ_ = θ + env.τ * θ_dot_
    end

    env.state = Tracker.data.([x_, ẋ_, θ_, θ_dot_])
    done =  !((-env.x_threshold ≤ x_ ≤ env.x_threshold) &&
            (-env.θ_threshold_radians ≤ θ_ ≤ env.θ_threshold_radians))

    if !done
        reward = 1f0
    elseif env.steps_beyond_done === nothing
        # Pole just fell!
        env.steps_beyond_done = 0
        reward = 1f0
    else
        if env.steps_beyond_done == 0
            @warn "You are calling 'step!()' even though this environment has already returned done = true. You should always call 'reset()' once you receive 'done = true' -- any further steps are undefined behavior."
        end
        env.steps_beyond_done += 1
        reward = 0f0
    end

    # Custom reward for training
    # Product of Triangular function over x-axis and θ-axis
    # Min reward = 0, Max reward = env.x_threshold * env.θ_threshold_radians
    r_x = max(0f0, min(env.x_threshold-x_, x_+env.x_threshold))
    r_θ = max(0f0, min(env.θ_threshold_radians-θ_, θ_+env.θ_threshold_radians))
    train_reward = r_x * r_θ
    return env.state, reward, done, train_reward, action
end

function reset!(env::CartPoleEnv)
    env.state = rand(Float32, 4) * 1f-1 .- 5f-2
    env.steps_beyond_done = nothing
    return env.state
end

show(io::IO, env::CartPoleEnv) = print(io, "CartPoleEnv")
