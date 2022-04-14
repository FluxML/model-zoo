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

    # Test our two use cases in function(unet::UNet)(x, t):
    reverse_test1 = randn(Float32, 32, 32)
    reverse_test2 = randn(Float32, 28, 28, 1, 32, 32)
    reverse_test3 = randn(Float32, 32)
    @assert reverse_dims(reverse_test1) == reverse_test1'
    # Array and Matrix
    @assert (
        reverse_test2 .+ expand_dims(reverse_test1, 3) ==
        reverse_dims(reverse_test2) .+ reverse_test1' |> reverse_dims
    )
    # Array and Vector
    @assert (
        reverse_test2 .+ expand_dims(reverse_test3, 4) ==
        reverse_dims(reverse_test2) .+ reverse_test3 |> reverse_dims
    )

    unet_test = UNet(marginal_prob_std)
    x_test = randn(Float32, (28, 28, 1, 32))
    t_test = rand(Float32, 32)
    score_test = unet_test(x_test, t_test)
    @assert score_test |> size == (28, 28, 1, 32)
    @assert typeof(score_test) == Array{Float32,4}
    # @time [unet_test(x_test, t_test) for i in 1:10];

    @info "Tests complete for diffusion_mnist.jl"
end

if abspath(PROGRAM_FILE) == @__FILE__
    test()
end