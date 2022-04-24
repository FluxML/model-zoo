include("diffusion_mnist.jl")

function test()
    @info "Begin tests for diffusion_mnist.jl"

    gaussfourierproj_test = GaussianFourierProjection(32, 20.0f0)
    # GaussianFourierProjection(embed_dim, ⋅)(batch) => (embed_dim, batch)
    @assert gaussfourierproj_test(randn(Float32, 32)) |> size == (32, 32)
    # W is fixed wrt. repeated calls
    @assert gaussfourierproj_test(
        ones(Float32, 32)) ==
            gaussfourierproj_test(ones(Float32, 32)
    )
    # W is not trainable
    @assert params(gaussfourierproj_test) == Flux.Params([])

    @assert expand_dims(ones(Float32, 32), 3) |> size == (1, 1, 1, 32)

    unet_test = UNet()
    x_test = randn(Float32, (28, 28, 1, 32))
    t_test = rand(Float32, 32)
    score_test = unet_test(x_test, t_test)
    @assert score_test |> size == (28, 28, 1, 32)
    @assert typeof(score_test) == Array{Float32,4}

    # Test gradient computation
    grad_test = gradient(
        () -> model_loss(unet_test, x_test), params(unet_test)
    )
    @assert grad_test.params == params(unet_test)

    train(save_path="test", epochs=1, batch_size=4096, tblogger=false)

    @info "Tests complete for diffusion_mnist.jl"
end

if abspath(PROGRAM_FILE) == @__FILE__
    test()
end