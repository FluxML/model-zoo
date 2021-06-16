# Conditional DCGAN

<img src="..\cdcgan_mnist\output\img_for_readme.png" width="440"/>

## Model Info

Generative Adversarial Networks have two models, a _Generator model G(z)_ and a _Discriminator model D(x)_, in competition with each other. G tries to estimate the distribution of the training data and D tries to estimate the probability that a data sample came from the original training data and not from G. During training, the Generator learns a mapping from a _prior distribution p(z)_ to the _data space G(z)_. The discriminator D(x) produces a probability value of a given x coming from the actual training data.
This model can be modified to include additional inputs, y, on which the models can be conditioned. y can be any type of additional inputs, for example, class labels. _The conditioning can be achieved by simply feeding y to both the Generator — G(z|y) and the Discriminator — D(x|y)_.

## Training

```shell
cd vision/cdcgan_mnist
julia --project cGAN_mnist.jl
```

## Results

1000 training step

![1000 training step](../cdcgan_mnist/output/cgan_steps_001000.png)

3000 training step

![30000 trainig step](../cdcgan_mnist/output/cgan_steps_003000.png)

5000 training step

![5000 training step](../cdcgan_mnist/output/cgan_steps_005000.png)

10000 training step

![10000 training step](../cdcgan_mnist/output/cgan_steps_010000.png)

11725 training step

![11725 training step](../cdcgan_mnist/output/cgan_steps_011725.png)

## References

[Conditional Generative Adversarial Nets by Mehdi Mirza et al.](https://arxiv.org/pdf/1411.1784.pdf)

[Medium](https://medium.com/@utk.is.here/training-a-conditional-dc-gan-on-cifar-10-fce88395d610)