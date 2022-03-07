# AlexNet

<p align="center">
  <img src="docs/AlexNet.png" />
</p>

AlexNet, proposed by Alex Krizhevsky and colleagues in 2012, partecipated in the ImageNet Large Scale Visual Recognition Challenge, obtaining outstanding results: top-5 error of 15.3%. It presents five convolutional layers and three fully-connected layers. One of the main contributions of this architecture was a first use of the ReLU activation function rather than the old well established tanh. Moreover, for the first time an architecture was able to support multi-GPU training, making the training faster. Lastly, in order to overcome overfitting, two other solutions were applied: Dropout (randomly inactivation of neurons) and Data Augumentation.

> Source : [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

### Train
```julia
cd vision/alexnet_cifar10
julia alexnet_cifar10.jl
```

### References
- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [Pytorch implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py)
