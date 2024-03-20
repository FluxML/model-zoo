include("vae_mnist.jl")

using Plots

function plot_result()
    checkpoint = JLD2.load("output/checkpoint.jld2")
    encoder_state = checkpoint["encoder"]
    decoder_state = checkpoint["decoder"]
    args = Args(; checkpoint["args"]...)
    encoder = Encoder(args.input_dim, args.latent_dim, args.hidden_dim)
    decoder = Decoder(args.input_dim, args.latent_dim, args.hidden_dim)
    Flux.loadmodel!(encoder, encoder_state)
    Flux.loadmodel!(decoder, decoder_state)
    loader = get_data(args.batch_size)

    # clustering in the latent space
    # visualize first two dims
    plt = scatter(palette=:rainbow)
    for (i, (x, y)) in enumerate(loader)
        i < 20 || break
        μ, logσ = encoder(x)
        @assert size(μ, 1) == 2 # Latent_dim has to be 2 for direct visualization, otherwise use PCA or t-SNE
        scatter!(μ[1, :], μ[2, :], 
            markerstrokewidth=0, markeralpha=0.8,
            aspect_ratio=1,
            markercolor=y, label="")
    end
    savefig(plt, "output/clustering.png")

    z = range(-2.0, stop=2.0, length=11)
    len = Base.length(z)
    z1 = repeat(z, len)
    z2 = sort(z1)
    x = zeros(Float32, args.latent_dim, len^2)
    x[1, :] = z1
    x[2, :] = z2
    samples = decoder(x)
    samples = sigmoid.(samples)
    image = convert_to_image(samples, len)
    save("output/manifold.png", image)
end

if abspath(PROGRAM_FILE) == @__FILE__ 
    plot_result()
end
