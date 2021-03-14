# Flux Model Zoo

This repository contains various demonstrations of the [Flux](http://fluxml.github.io/) machine learning library. Any of these may freely be used as a starting point for your own models.

The models are broadly categorised into the folders [vision](/vision) (e.g. large convolutional neural networks (CNNs)), [text](/text) (e.g. various recurrent neural networks (RNNs) and natural language processing (NLP) models), [games](/contrib/games) (Reinforcement Learning / RL). See the READMEs of respective models for more information.

## Usage

The zoo comes with its own [Julia project](https://julialang.github.io/Pkg.jl/latest/#Using-someone-else's-project-1), which lists the packages you need to run the models. You can run the models by opening Julia in the project folder and running

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
```

to install all needed packages. Then you can run the model code with `include("script.jl")` or by running the script line-by-line.

Models may also be run with NVIDIA GPU support, if you have a CUDA installed. Most models will have this capability by default, pointed at by calls to `gpu` in the model code.

### Gitpod Online IDE

Each model can be used in [Gitpod](https://www.gitpod.io/), just [open the repository by gitpod](https://gitpod.io/#https://github.com/FluxML/model-zoo)

* Based on [Gitpod's policies](https://www.gitpod.io/pricing/), free access is limited.
* All of your work will place in the Gitpod's cloud.
* It isn't an officially maintained feature.

## Contributing

We welcome contributions of new models. They should be in a folder with a project and manifest file, to pin all relevant packages, as well as a README to explain what the model is about, how to run it, and what results it achieves (if applicable). If possible models should not depend directly on GPU functionality, but ideally should be CPU/GPU agnostic. Please keep the code short, clean and self-explanatory, with as little boilerplate code as possible.

## Examples Listing

* Vision
  * MNIST
    * [Simple multi-layer perceptron](vision/mlp_mnist)
    * [Simple ConvNet (LeNet)](vision/conv_mnist)
    * [Variational Auto-Encoder](vision/vae_mnist)
    * [Deep Convolutional Generative Adversarial Networks](vision/dcgan_mnist)
    * [Conditional Deep Convolutional Generative Adversarial Networks](vision/cdcgan_mnist)
  * [VGG 16/19 on CIFAR10](vision/vgg_cifar10)
* Text
  * [CharRNN](text/char-rnn)
  * [Character-level language detection](text/lang-detection)
  * [Seq2Seq phoneme detection on CMUDict](text/phonemes)
  * [Recursive net on IMDB sentiment treebank](text/treebank)
* Other & contributed models
  * [Differential Equations](https://diffeqflux.sciml.ai/dev)
  * [BitString Parity Challenge](other/bitstring-parity)
  * [MLP on housing data](other/housing/housing.jl) (low level API)
  * [FizzBuzz](other/fizzbuzz/fizzbuzz.jl)
  * [Meta-Learning](contrib/meta-learning/MetaLearning.jl)
  * [Logistic Regression Iris](other/iris/iris.jl)
  * [Speech recognition](contrib/audio/speech-blstm)
* Tutorials
  * [A 60 Minute Blitz](tutorials/60-minute-blitz/60-minute-blitz.jl)
  * [DataLoader example with image data](tutorials/dataloader)
  * [Transfer Learning](tutorials/transfer_learning/transfer_learning.jl)
