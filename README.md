# Flux Model Zoo

**Note:** These examples are in the process of being upgraded for Julia 1.0. Until then, they work best with Julia v0.6.

This repository contains various demonstrations of the [Flux](http://fluxml.github.io/) machine learning library. Any of these may freely be used as a starting point for your own models.

The models are broadly categorised into the folders [vision](/vision) (e.g. large convnets), [text](/text) (e.g. various RNNs and NLP models), [games](/games) (reinforcement learning). See the READMEs of respective models for more information.

## Usage

Each folder is its own [Julia project](https://julialang.github.io/Pkg.jl/latest/#Using-someone-else's-project-1), which lists the packages you need to run the models. You can run the models by opening Julia in the project folder and running

```
]activate .; instantiate
```

to install all needed packages. Then you can run the model code with `include("script.jl")` or by running the script line-by-line. More details are available in the README for each model.

## Contributing

We welcome contributions of new models. They should be in a folder with a project and manifest file, to pin all relevant packages, as well as a README to explain what the model is about, how to run it, and what results it acheives (if applicable). If possible models should not depend directly on GPU functionality, but ideally should be CPU/GPU agnostic.

## Model Listing

* Vision
  * MNIST
    * [Simple multi-layer perceptron](vision/mnist/mlp.jl)
    * [Simple ConvNet](vision/mnist/conv.jl)
    * [Simple Auto-Encoder](vision/mnist/autoencoder.jl)
    * [Variational Auto-Encoder](vision/mnist/vae.jl)
  * [VGG 16/19 on CIFAR10](vision/cifar10)
* Text
  * [CharRNN](text/char-rnn)
  * [Character-level language detection](text/lang-detection)
  * [Seq2Seq phoneme detection on CMUDict](text/phonemes)
  * [Recursive net on IMDB sentiment treebank](text/treebank)
* Other
  * [BitString Parity Challenge](other/bitstring-parity)
  * [MLP on housing data](other/housing) (low level API)
