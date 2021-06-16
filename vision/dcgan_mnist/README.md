# Deep Convolutional GAN

![dcgan_gen_disc](../dcgan_mnist/docs/dcgan_generator_discriminator.png)

## Model Info

A DCGAN is a direct extension of the GAN, except that it explicitly uses convolutional and convolutional-transpose layers in the discriminator and generator, respectively. _The discriminator is made up of strided convolution layers, batch norm layers, and LeakyReLU activations_. The input is a 3x64x64 input image and the output is a scalar probability that the input is from the real data distribution. _The generator is comprised of convolutional-transpose layers, batch norm layers, and ReLU activations_. The input is a latent vector, _z_, that is drawn from a standard normal distribution and the output is a 3x64x64 RGB image. The strided conv-transpose layers allow the latent vector to be transformed into a volume with the same shape as an image.

## Training

```script
cd vision/dcgan_mnist
julia --project dcgan_mnist.jl
```

## Results

2000 training step

![2000 training steps](../dcgan_mnist/output/dcgan_steps_002000.png)

5000 training step

![5000 training steps](../dcgan_mnist/output/dcgan_steps_005000.png)

8000 training step

![8000 training steps](../dcgan_mnist/output/dcgan_steps_008000.png)

9380 training step

![9380 training step](../dcgan_mnist/output/dcgan_steps_009380.png)

## References

[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks by Soumith Chintala et al.](https://arxiv.org/pdf/1511.06434v2.pdf)

[pytorch.org/tutorials/beginner/dcgan_faces_tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)