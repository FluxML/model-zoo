<img align="right" width="200px" src="https://github.com/FluxML/Optimisers.jl/raw/master/docs/src/assets/logo.png">

# Flux Model Zoo

This repository contains various demonstrations of the [Flux](http://fluxml.github.io/) machine learning library. Any of these may freely be used as a starting point for your own models.

The models are broadly categorised into the folders [vision](/vision) (e.g. large convolutional neural networks (CNNs)), [text](/text) (e.g. various recurrent neural networks (RNNs) and natural language processing (NLP) models), [games](/contrib/games) (Reinforcement Learning / RL). See the READMEs of respective models for more information.

## Usage

The zoo comes with its own [Julia project](https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project), which lists the packages you need to run the models. You can run the models by opening Julia in the project folder and running

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
```

to install all needed packages. Then you can run the model code with `include("<model-to-run>.jl")` or by running the model script line-by-line.

Models may also be run with NVIDIA GPU support, if you have a CUDA installed. Most models will have this capability by default, pointed at by calls to `gpu` in the model code.

### Gitpod Online IDE

Each model can be used in [Gitpod](https://www.gitpod.io/), just [open the repository by gitpod](https://gitpod.io/#https://github.com/FluxML/model-zoo)

* Based on [Gitpod's policies](https://www.gitpod.io/pricing/), free access is limited.
* All of your work will place in the Gitpod's cloud.
* It isn't an officially maintained feature.

## Contributing

We welcome contributions of new models and documentation. 

### Share a new model

If you want to share a new model, we suggest you follow these guidelines:

* Models should be in a folder with a project and manifest file to pin all relevant packages.
* Models should include a README(.md) to explain what the model is about, how to run it, and what results it achieves (if applicable).
* Models should ideally be CPU/GPU agnostic and not depend directly on GPU functionality.
* Please keep the code short, clean, and self-explanatory, with as little boilerplate as possible.

### Create or improve documentation

You can contribute in one of the following ways 

* Add or improve documentation to existing models: Write the following information:
  * Give a brief introduction to the model’s architecture and the goal it archives.
  * Describe the Flux API that the model demonstrates (high-level API, AD, custom operations, custom layers, etc.).
  * Add literature background for the model. More specifically, add articles, blog posts, videos, and any other resource that is helpful to better understand the model.
  * Mention the technique that is being demonstrated. Briefly describe the learning technique being demonstrated (Computer vision, regression, NLP, time series, etc.).
* Write in-depth tutorials for a model: You can further extend the documentation of a model and create a tutorial to explain in more detail the architecture, the training routine, use your own data, and so forth. After you write a tutorial, create a PR with it for the [Tutorials](https://fluxml.ai/tutorials.html) section on the [FluxML](https://fluxml.ai/) website.

### Update a model

Each example lists the version of Flux for which it was most recently updated.
Bringing them up to the latest is a great way to learn!
Flux has a [NEWS page](https://github.com/FluxML/Flux.jl/blob/master/NEWS.md) listing important changes.
(For other packages, see their releses page: [MLUtils](https://github.com/JuliaML/MLUtils.jl/releases), [MLDatasets](https://github.com/JuliaML/MLDatasets.jl/releases), etc.)

To run the old examples, Flux v0.11 can be installed and run on [Julia 1.6, the LTS version](https://julialang.org/downloads/#long_term_support_release).
Flux v0.12 works on Julia 1.8.
Flux 0.13 is the latest right now, marked with ☀️.

## Examples Listing

**Vision**
* MNIST
  * [Simple multi-layer perceptron](vision/mlp_mnist) ☀️ v0.13
  * [Simple ConvNet (LeNet)](vision/conv_mnist) ☀️ v0.13
  * [Variational Auto-Encoder](vision/vae_mnist) 🌤 v0.12
  * [Deep Convolutional Generative Adversarial Networks](vision/dcgan_mnist) ☀️ v0.13
  * [Conditional Deep Convolutional Generative Adversarial Networks](vision/cdcgan_mnist) ☀️ v0.13
  * [Score-Based Generative Modeling (Diffusion Model)](vision/diffusion_mnist) ☀️ v0.13
  * [Spatial Transformer](vision/spatial_transformer) ☀️ v0.13
* CIFAR10
  * [VGG 16/19](vision/vgg_cifar10) ☀️ v0.13
  * [ConvMixer "Patches are all you need?"](vision/convmixer_cifar10/) ☀️ v0.13

**Text**
* [CharRNN](text/char-rnn) ⛅️ v0.11
* [Character-level language detection](text/lang-detection) ⛅️ v0.11
* [Seq2Seq phoneme detection on CMUDict](text/phonemes) ⛅️ v0.11
* [Recursive net on IMDB sentiment treebank](text/treebank) ⛅️ v0.11

**Other** & contributed models
* [Logistic Regression Iris](other/iris/iris.jl) ☀️ v0.13
* [Autoregressive Model](other/autoregressive-process/) ☀️ v0.13
* [BitString Parity Challenge](other/bitstring-parity) ⛅️ v0.11
* [MLP on housing data](other/housing/) (low level API) ⛅️ v0.11
* [FizzBuzz](other/fizzbuzz/fizzbuzz.jl) ⛅️ v0.11
* [Meta-Learning](contrib/meta-learning/MetaLearning.jl) ❄️ v0.7
* [Speech recognition](contrib/audio/speech-blstm) ❄️ v0.6

**Tutorials**
* [A 60 Minute Blitz](tutorials/60-minute-blitz/60-minute-blitz.jl) ⛅️ v0.11
* [DataLoader example with image data](tutorials/dataloader) ⛅️ v0.11
* [Transfer Learning](tutorials/transfer_learning/transfer_learning.jl) ⛅️ v0.11

