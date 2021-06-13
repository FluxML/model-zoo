# Variational Autoencoder (VAE)

<img src="..\vae_mnist\docs\vae.png" width="500">

## Training

```shell
cd vision/vae_mnist
julia --project vae_mnist.jl
```

Original image

![Original](docs/original.png)

5 epochs

![5 epochs](docs/epoch_5.png)

10 epochs

![10 epochs](docs/epoch_10.png)

20 epochs

![10 epochs](docs/epoch_20.png)

## Visualization

```shell
julia --project vae_plot.jl
```

Latent space clustering

![Clustering](docs/clustering.png)

2D manifold

![Manifold](docs/manifold.png)
