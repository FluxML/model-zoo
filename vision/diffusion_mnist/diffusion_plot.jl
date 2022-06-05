include("diffusion_mnist.jl")
using Images
using ProgressMeter
using DifferentialEquations
using Plots


"""
Helper function yielding the diffusion coefficient from a SDE.
"""
diffusion_coeff(t, sigma=convert(eltype(t), 25.0f0)) = sigma .^ t

"""
Helper function that produces images from a batch of images.
"""
function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(x |> cpu, y_size), 28, :)...), (2, 1)))
end

"""
Helper to make an animation from a batch of images.
"""
function convert_to_animation(x)
    frames = size(x)[end]
    batches = size(x)[end-1]
    animation = @animate for i = 1:frames+frames÷4
        if i <= frames
            heatmap(
                convert_to_image(x[:, :, :, :, i], batches),
                title="Iteration: $i out of $frames"
            )
        else
            heatmap(
                convert_to_image(x[:, :, :, :, end], batches),
                title="Iteration: $frames out of $frames"
            )
        end
    end
    return animation
end

"""
Helper function that generates inputs to a sampler.
"""
function setup_sampler(device, num_images=5, num_steps=500, ϵ=1.0f-3)
    t = ones(Float32, num_images) |> device
    init_x = (
        randn(Float32, (28, 28, 1, num_images)) .*
        expand_dims(marginal_prob_std(t), 3)
    ) |> device
    time_steps = LinRange(1.0f0, ϵ, num_steps)
    Δt = time_steps[1] - time_steps[2]
    return time_steps, Δt, init_x
end

"""
Sample from a diffusion model using the Euler-Maruyama method.

# References
https://yang-song.github.io/blog/2021/score/#how-to-solve-the-reverse-sde
"""
function Euler_Maruyama_sampler(model, init_x, time_steps, Δt)
    x = mean_x = init_x
    @showprogress "Euler-Maruyama Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
        g = diffusion_coeff(batch_time_step)
        mean_x = x .+ expand_dims(g, 3) .^ 2 .* model(x, batch_time_step) .* Δt
        x = mean_x .+ sqrt(Δt) .* expand_dims(g, 3) .* randn(Float32, size(x))
    end
    return mean_x
end

"""
Sample from a diffusion model using the Predictor-Corrector method.

# References
https://yang-song.github.io/blog/2021/score/#how-to-solve-the-reverse-sde
"""
function predictor_corrector_sampler(model, init_x, time_steps, Δt, snr=0.16f0)
    x = mean_x = init_x
    @showprogress "Predictor Corrector Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
        # Corrector step (Langevin MCMC)
        grad = model(x, batch_time_step)
        num_pixels = prod(size(grad)[1:end-1])
        grad_batch_vector = reshape(grad, (size(grad)[end], num_pixels))
        grad_norm = mean(sqrt, sum(abs2, grad_batch_vector, dims=2))
        noise_norm = Float32(sqrt(num_pixels))
        langevin_step_size = 2 * (snr * noise_norm / grad_norm)^2
        x += (
            langevin_step_size .* grad .+
            sqrt(2 * langevin_step_size) .* randn(Float32, size(x))
        )
        # Predictor step (Euler-Maruyama)
        g = diffusion_coeff(batch_time_step)
        mean_x = x .+ expand_dims((g .^ 2), 3) .* model(x, batch_time_step) .* Δt
        x = mean_x + sqrt.(expand_dims((g .^ 2), 3) .* Δt) .* randn(Float32, size(x))
    end
    return mean_x
end

"""
Helper to create a SDEProblem with DifferentialEquations.jl

# Notes
The reverse-time SDE is given by:  
𝘥x = -σ²ᵗ 𝘚₀(𝙭, 𝘵)𝘥𝘵 + σᵗ𝘥𝘸  
⟹ `f(u, p, t)` = -σ²ᵗ 𝘚₀(𝙭, 𝘵)  
⟹ `g(u, p, t` = σᵗ
"""
function DifferentialEquations_problem(model, init_x, time_steps, Δt)
    function f(u, p, t)
        batch_time_step = fill!(similar(u, size(u)[end]), 1) .* t
        return (
            -expand_dims(diffusion_coeff(batch_time_step), 3) .^ 2 .*
            model(u, batch_time_step)
        )
    end

    function g(u, p, t)
        batch_time_step = fill!(similar(u), 1) .* t
        diffusion_coeff(batch_time_step)
    end
    tspan = (time_steps[begin], time_steps[end])
    SDEProblem(f, g, init_x, tspan), ODEProblem(f, init_x, tspan)
end

function plot_result(unet, args)
    args = Args(; args...)
    args.seed > 0 && Random.seed!(args.seed)
    device = args.cuda && CUDA.has_cuda() ? gpu : cpu
    unet = unet |> device
    time_steps, Δt, init_x = setup_sampler(device)

    # Euler-Maruyama
    euler_maruyama = Euler_Maruyama_sampler(unet, init_x, time_steps, Δt)
    sampled_noise = convert_to_image(init_x, size(init_x)[end])
    save(joinpath(args.save_path, "sampled_noise.jpeg"), sampled_noise)
    em_images = convert_to_image(euler_maruyama, size(euler_maruyama)[end])
    save(joinpath(args.save_path, "em_images.jpeg"), em_images)

    # Predictor Corrector
    pc = predictor_corrector_sampler(unet, init_x, time_steps, Δt)
    pc_images = convert_to_image(pc, size(pc)[end])
    save(joinpath(args.save_path, "pc_images.jpeg"), pc_images)

    # Setup an SDEProblem and ODEProblem to input to `solve()`.
    # Use dt=Δt to make the sample paths comparable to calculating "by hand".
    sde_problem, ode_problem = DifferentialEquations_problem(unet, init_x, time_steps, Δt)

    @info "Euler-Maruyama Sampling w/ DifferentialEquations.jl"
    diff_eq_em = solve(sde_problem, EM(), dt=Δt)
    diff_eq_em_end = diff_eq_em[:, :, :, :, end]
    diff_eq_em_images = convert_to_image(diff_eq_em_end, size(diff_eq_em_end)[end])
    save(joinpath(args.save_path, "diff_eq_em_images.jpeg"), diff_eq_em_images)
    diff_eq_em_animation = convert_to_animation(diff_eq_em)
    gif(diff_eq_em_animation, joinpath(args.save_path, "diff_eq_em.gif"), fps=50)
    em_plot = plot(diff_eq_em, title="Euler-Maruyama", legend=false, ylabel="x", la=0.25)
    plot!(time_steps, diffusion_coeff(time_steps), xflip=true, ls=:dash, lc=:red)
    plot!(time_steps, -diffusion_coeff(time_steps), xflip=true, ls=:dash, lc=:red)
    savefig(em_plot, joinpath(args.save_path, "diff_eq_em_plot.png"))

    @info "Probability Flow ODE Sampling w/ DifferentialEquations.jl"
    diff_eq_ode = solve(ode_problem, dt=Δt, adaptive=false)
    diff_eq_ode_end = diff_eq_ode[:, :, :, :, end]
    diff_eq_ode_images = convert_to_image(diff_eq_ode_end, size(diff_eq_ode_end)[end])
    save(joinpath(args.save_path, "diff_eq_ode_images.jpeg"), diff_eq_ode_images)
    diff_eq_ode_animation = convert_to_animation(diff_eq_ode)
    gif(diff_eq_ode_animation, joinpath(args.save_path, "diff_eq_ode.gif"), fps=50)
    ode_plot = plot(diff_eq_ode, title="Probability Flow ODE", legend=false, ylabel="x", la=0.25)
    plot!(time_steps, diffusion_coeff(time_steps), xflip=true, ls=:dash, lc=:red)
    plot!(time_steps, -diffusion_coeff(time_steps), xflip=true, ls=:dash, lc=:red)
    savefig(ode_plot, joinpath(args.save_path, "diff_eq_ode_plot.png"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    ############################################################################
    # Issue loading function closures with BSON:
    # https://github.com/JuliaIO/BSON.jl/issues/69
    #
    BSON.@load "output/model.bson" unet args
    #
    # BSON.@load does not work if defined inside plot_result(⋅) because
    # it contains a function closure, GaussFourierProject(⋅), containing W.
    ###########################################################################
    plot_result(unet, args)
end
