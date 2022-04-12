# include("vision/diffusion_mnist/diffusion_mnist.jl")
include("diffusion_mnist.jl")

using Plots

function diffusion_coeff(t, sigma=25.0f0)
    sigma .^ t
end

function convert_to_image(x, y_size)
    Gray.(
        permutedims(
            vcat(
                reshape.(
                    chunk(x |> cpu, y_size), 28, :
                )...
            ),
            (2, 1)
        )
    )
end

"""
Sample from a diffusion model using the Euler-Maruyama method

# References
https://yang-song.github.io/blog/2021/score/#how-to-solve-the-reverse-sde
"""
function Euler_Maruyama_sampler(model, diffusion_coeff, num_images, device, num_steps=500, ϵ=1.0f-3)
    t = ones(Float32, num_images) |> device
    init_x = (
        randn(Float32, (28, 28, 1, num_images)) .*
        expand_dims(model.marginal_prob_std(t), 3)
    ) |> device
    time_steps = LinRange(1.0f0, ϵ, num_steps)
    Δt = time_steps[1] - time_steps[2]
    x = init_x
    progress = Progress(length(time_steps))
    @info "Start Sampling, total $(num_steps) steps"
    for time_step in time_steps
        batch_time_step = ones(Float32, num_images) * time_step |> device
        g = diffusion_coeff(batch_time_step)
        mean_x = x .+ expand_dims((g .^ 2), 3) .* model(x, batch_time_step) .* Δt
        x = mean_x + sqrt(Δt) * expand_dims(g, 3) .* randn(Float32, size(x))
        next!(progress; showvalues=[(:time_step, time_step)])
    end
    return init_x, mean_x
end

function plot_result(num_images=5)
    BSON.@load "vision/diffusion_mnist/output/model.bson" unet args
    args = Args(; args...)
    device = args.cuda && CUDA.has_cuda() ? gpu : cpu
    unet = unet |> device
    init_x, mean_x = Euler_Maruyama_sampler(unet, diffusion_coeff, num_images, device)
    sampled_noise = convert_to_image(init_x[:, :, :, 1:num_images], num_images)
    save(joinpath(args.save_path, "sampled_noise.png"), sampled_noise)
    sampled_images = convert_to_image(mean_x[:, :, :, 1:num_images], num_images)
    save(joinpath(args.save_path, "sampled_images.png"), sampled_images)
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_result()
end
