# # Transfer Learning with Flux

# This article is intended to be a general guide to how transfer learning works in the Flux ecosystem.
# We assume a certain familiarity of the reader with the concept of transfer learning. Having said that,
# we will start off with a basic definition of the setup and what we are trying to achieve. There are many
# resources online that go in depth as to why transfer learning is an effective tool to solve many ML
# problems, and we recommend checking some of those out.

# Machine Learning today has evolved to use many highly trained models in a general task,
# where they are tuned to perform especially well on a subset of the problem.

# This is one of the key ways in which larger (or smaller) models are used in practice. They are trained on
# a general problem, achieving good results on the test set, and then subsequently tuned on specialised datasets.

# In this process, our model is already pretty well trained on the problem, so we don't need to train it
# all over again as if from scratch. In fact, as it so happens, we don't need to do that at all! We only
# need to tune the last couple of layers to get the most performance from our models. The exact last number of layers
# is dependant on the problem setup and the expected outcome, but a common tip is to train the last few `Dense`
# layers in a more complicated model.

# So let's try to simulate the problem in Flux.

# We'll tune a pretrained ResNet from Metalhead as a proxy. We will tune the `Dense` layers in there on a new set of images.

using Flux, Metalhead
using Flux: @epochs
resnet = ResNet().layers

# If we intended to add a new class of objects in there, we need only `reshape` the output from the previous layers accordingly.
# Our model would look something like so:

# ```julia
# model = Chain(
#   resnet[1:end-2],               # We only need to pull out the dense layer in here
#   x -> reshape(x, size_we_want), # / global_avg_pooling layer
#   Dense(reshaped_input_features, n_classes)
# )
# ```

# We will use the [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) dataset from Kaggle for our use here.
# Make sure to extract the images in a `train` folder.

# The `datatloader.jl` script contains some functions that will help us load batches of images, shuffled between
# dogs and cats along with their correct labels.

include("dataloader.jl")

# Finally, the model looks something like:

model = Chain(
  resnet[1:end-2],
  Dense(2048, 1000),  
  Dense(1000, 256),
  Dense(256, 2),        # we get 2048 features out, and we have 2 classes
)

# To speed up training, let’s move everything over to the GPU

model = model |> gpu
dataset = [gpu.(load_batch(10)) for i in 1:10]

# After this, we only need to define the other parts of the training pipeline like we usually do.

opt = ADAM()
loss(x,y) = Flux.Losses.logitcrossentropy(model(x), y)

# Now to train
# As discussed earlier, we don’t need to pass all the parameters to our training loop. Only the ones we need to
# fine-tune. Note that we could have picked and chosen the layers we want to train individually as well, but this
# is sufficient for our use as of now.

ps = Flux.params(model[2:end])  # ignore the already trained layers of the ResNet

# And now, let's train!

@epochs 2 Flux.train!(loss, ps, dataset, opt)

# And there you have it, a pretrained model, fine tuned to tell the the dogs from the cats.

# We can verify this too.

imgs, labels = gpu.(load_batch(10))
display(model(imgs))

labels

