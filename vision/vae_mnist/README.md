# Variational Autoencoder (VAE)

<img src="..\vae_mnist\docs\vae.png" width="500">

## Model Info

Variational Autoencoder ( VAE ) came into existence in 2013, when Diederik et al. published a paper [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf). This paper was an extension of the original idea of Auto-Encoder primarily to learn the useful distribution of the data. Variational Autoencoder was inspired by the methods of the variational bayesian and graphical model. VAE is rooted in Bayesian inference, i.e., it wants to model the underlying probability distribution of data to sample new data from that distribution.

VAE has one fundamentally unique property that separates them from vanilla autoencoder, and it is this property that makes them so useful for generative modeling: their latent spaces are, by design, continuous, allowing easy random sampling and interpolation.

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

## Reference

[learnopencv](https://learnopencv.com/variational-autoencoder-in-tensorflow/#:~:text=Variational%20Autoencoder%20%28VAE%29%20is%20a%20generative%20model%20that,Gaussian%20profile%20%28prior%20on%20the%20distribution%20of%20representations%29.)