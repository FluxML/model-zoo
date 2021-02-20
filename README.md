# Flux Model Zoo

This repository contains various demonstrations of the [Flux](http://fluxml.github.io/) machine learning library. Any of these may freely be used as a starting point for your own models.

The models are broadly categorised into the folders [vision](/vision) (e.g. convolutional neural networks (CNNs)), [text](/text) (e.g. various recurrent neural networks (RNNs) and natural language processing (NLP) models), and [other](/contrib/other).

## Usage

The zoo's examples comes with their own [Julia project](https://julialang.github.io/Pkg.jl/dev/environments/#Using-someone-else's-project), which lists the packages you need to run the models. You can run the models by opening Julia in the project folder and running
(mixed stuff such as the Iris dataset)

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
```

to install all needed packages. Then you can run the model code with `include("script.jl)` or by running the script line-by-line.

Models may also be run with NVIDIA GPU support, if you have a CUDA installed. Most models will have this capability by default, pointed at by calls to `gpu` in the model code.

### Gitpod Online IDE

Each model can be used in [Gitpod](https://www.gitpod.io/), just [open the repository by gitpod](https://gitpod.io/#https://github.com/FluxML/model-zoo)

#### Considerations

* Based on [Gitpod's policies](https://www.gitpod.io/pricing/), free access is limited.
* All of your work will place in the Gitpod's cloud.
* It isn't an officially maintained feature.

## Contributing

We welcome contributions of new models. They should be in a folder with a project and manifest file, to pin all relevant packages, as well as a README to explain what the model is about, how to run it, and what results it achieves (if applicable). If possible models should not depend directly on GPU functionality, but ideally should be CPU/GPU agnostic.

## Model Listing

* Vision
  * MNIST
    * [Simple multi-layer perceptron](vision/mlp_mnist/)
    * [Simple ConvNet (LeNet)](vision/conv_mnist/)
    * [Variational Auto-Encoder](vision/vae_mnist/)
    * [Deep Convolutional Generative Adversarial Networks](vision/dcgan_mnist/)
    * [Conditional Deep Convolutional Generative Adversarial Networks](vision/cdcgan_mnist/)
  * [VGG 16/19 on CIFAR10](vision/vgg_cifar10)
* Text
  * [CharRNN](text/char-rnn)
  * [Character-level language detection](text/lang-detection)
  * [Seq2Seq phoneme detection on CMUDict](text/phonemes)
  * [Recursive net on IMDB sentiment treebank](text/treebank)
* Other & contributed models
  * [Differential Equations](https://diffeqflux.sciml.ai/dev)
  * [Reinforcement Learning](https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl)
  * [BitString Parity Challenge](other/bitstring-parity)
  * [MLP on housing data](other/housing/) (low level API)
  * [FizzBuzz](other/fizzbuzz/)
  * [Logistic Regression Iris](other/iris/)
  * [Speech recognition](contrib/audio/speech-blstm)
* Tutorials
  * [A 60 Minute Blitz](tutorials/60-minute-blitz/)
  * [DataLoader example with image data](tutorials/dataloader)
  * [Transfer Learning](tutorials/transfer_learning/)
