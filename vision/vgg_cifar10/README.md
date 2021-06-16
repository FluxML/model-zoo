# Visual Geometry Group (VGG)

![vgg](../vgg_cifar10/docs/vgg.png)

## Model Info

The basic building block of classic CNNs is a sequence of the following: (i) a convolutional layer with padding to maintain the resolution, (ii) a nonlinearity such as a ReLU, (iii) a pooling layer such as a maximum pooling layer. _One VGG block consists of a sequence of convolutional layers, followed by a maximum pooling layer for spatial downsampling_. In the original VGG paper [Simonyan & Zisserman, 2014](https://arxiv.org/pdf/1409.1556v4.pdf), the authors employed convolutions with  3×3  kernels with padding of 1 (keeping height and width) and  2×2  maximum pooling with stride of 2 (halving the resolution after each block).

## Training

```shell
cd vision/vgg_cifar10
julia --project vgg_cifar10.jl
```

## Reference

[d2l.ai](http://d2l.ai/chapter_convolutional-modern/vgg.html)