# Flux Model Zoo

This repository contains various demonstrations of the [Flux](http://fluxml.github.io/) machine learning library. Any of these may freely be used as a starting point for your own models.

The models are broadly categorised into the folders [vision](/vision) (e.g. large convolutional neural networks (CNNs)), [text](/text) (e.g. various recurrent neural networks (RNNs) and natural language processing (NLP) models), [games](/games) (Reinforcement Learning / RL). See the READMEs of respective models for more information.

## Usage

Each folder is its own [Julia project](https://julialang.github.io/Pkg.jl/latest/#Using-someone-else's-project-1), which lists the packages you need to run the models. You can run the models by opening Julia in the project folder and running

```
using Pkg; Pkg.activate("."); Pkg.instantiate()
```

to install all needed packages. Then you can run the model code with `include("script.jl")` or by running the script line-by-line. More details are available in the README for each model.

Models with a `cuda` folder can be loaded with NVIDIA GPU support, if you have a CUDA installed.

```julia
using Pkg; Pkg.activate("cuda"); Pkg.instantiate()
using CuArrays
```
Note: Place your downloaded dataset where MetalHead is installed, typically in `/home/<user>/.julia/packages/MetalHead/fYeSU/src/datasets`

## Contributing

We welcome contributions of new models. They should be in a folder with a project and manifest file, to pin all relevant packages, as well as a README to explain what the model is about, how to run it, and what results it achieves (if applicable). If possible models should not depend directly on GPU functionality, but ideally should be CPU/GPU agnostic.

## Model Listing

* Vision
  * MNIST
    * [Simple multi-layer perceptron](vision/mnist/mlp.jl)
    * [Simple ConvNets](vision/mnist/conv.jl)
    * [Simple Auto-Encoder](vision/mnist/autoencoder.jl)
    * [Variational Auto-Encoder](vision/mnist/vae.jl)
  * [VGG 16/19 on CIFAR10](vision/cifar10)
  * [CPPN](vision/cppn) ([Blog](http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/))
* Text
  * [CharRNN](text/char-rnn)
  * [Character-level language detection](text/lang-detection)
  * [Seq2Seq phoneme detection on CMUDict](text/phonemes)
  * [Recursive net on IMDB sentiment treebank](text/treebank)
* Other
  * [Differential Equations](other/diffeq)
  * [BitString Parity Challenge](other/bitstring-parity)
  * [MLP on housing data](other/housing/housing.jl) (low level API)
  * [FizzBuzz](other/fizzbuzz/fizzbuzz.jl)
  * [Meta-Learning](other/meta-learning/MetaLearning.jl)
  * [Logistic Regression Iris](other/iris/iris.jl)
