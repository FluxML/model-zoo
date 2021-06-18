# Multilayer Perceptron (MLP)

![mlp](../mlp_mnist/docs/mlp.svg)
[Source](http://d2l.ai/chapter_multilayer-perceptrons/mlp.html)

## Model Info

An MLP consists of at least three of nodes: an input layer, a hidden layer and an output layer. Except for the input node each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.

## Training

```script
cd vision/mlp_mnist
julia --project mlp_mnist.jl
```

## Reference

* [Multilayer Perceptrons](http://d2l.ai/chapter_multilayer-perceptrons/mlp.html)
