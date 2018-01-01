# Flux Model Zoo

This repository contains various demonstrations of the [Flux](http://fluxml.github.io/) machine learning library. Any of these may freely be used as a starting point for your own models.

- **housing** implements the most basic model possible (a linear regression) on the [UCI housing data set](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/). It's bare-bones and illustrates how to build a model from scratch.
- **mnist** classifies digits from the [MNIST data set](https://en.wikipedia.org/wiki/MNIST_database), using a simple multi-layer perceptron.
- **char-rnn** implements a [character-level language model](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). It comes with a Shakespeare dataset but can work with any text.
- **phonemes** implements a [sequence to sequence model with attention](https://arxiv.org/abs/1409.0473), using the [CMU pronouncing dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) to predict the pronunciations of unknown words.
- **action-recognition** implements [Large-scale Video Classification with Convolutional Neural Networks](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf)

Note that these models are best run line-by-line, either in the REPL or Juno.
