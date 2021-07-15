# Scene Category Dataset

Dataset Publicly Available at http://cvcl.mit.edu/MM/index.html

## Additional Dependencies

1. Images.jl
2. BSON.jl
3. CuArrays.jl
4. Augmentor.jl

## Explanation

The dataset is heavily biased and some classes have only one image. This
means we need to augment the given images to make a more balanced
dataset.

Firstly, we need to generate images of dimension 224 x 224. So we
generate 5 crops from each image --- 4 from each corner and 1 centre
crop. Next we need to apply augmentation to each of these images.

The following Augmentation were applied to the images :-
* Horizontal Flipping
* Rotation of the image
* Elastic Distortion
* Gaussian Filter

For additional augmentation techniques lookup the documentation of
[Augmentor.jl](https://evizero.github.io/Augmentor.jl/) and [Images.jl](https://juliaimages.github.io/)

In the process of augmentation the dimensions of the image might be
altered. Hence, we resize all the images back to 224 x 224 dimension.

Lastly, split the images into Train and Validation Data. Validation Data
is made of 10 % of each of the categories.

`load_data.jl` file is responsible for getting the data into proper
format.

By default the model used is `ResNet18`. In order to change the model
change [this line](https://github.com/avik-pal/model-zoo/blob/6c99bb46fcefd6966b0c137c68a8c265941fb853/Scene%20Category/train.jl#L1). `ResNet` function supports the Standard ResNet models of depth 18, 34, 50, 101 and 152. If you are using a deeper model consider reducting the [batch size](https://github.com/avik-pal/model-zoo/blob/6c99bb46fcefd6966b0c137c68a8c265941fb853/Scene%20Category/load_data.jl#L50) as you might run out of memory. By using the default settings and training for 2 additional epochs with a learning rate of 0.01 a validation accuracy of 99.98% should be achieved.

Finally to start training the model execute `julia run.jl`. By default
the training will happen on GPU. So if you need to train on CPU comment
out the `using CuArrays` line in `run.jl` file.
