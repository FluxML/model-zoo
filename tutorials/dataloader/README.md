# Using Flux DataLoader with image data

In this example, we show how to load image data in Flux DataLoader and process it in mini-batches. We use the [DataLoader](https://fluxml.ai/Flux.jl/stable/data/dataloader/#Flux.Data.DataLoader) type to handle iteration over mini-batches of data. For this example, we load the [MNIST dataset](https://juliaml.github.io/MLDatasets.jl/latest/datasets/MNIST/) using the [MLDatasets](https://juliaml.github.io/MLDatasets.jl/latest/) package.
 
Before we start, make sure you have installed the following packages:
 
* [Flux](https://github.com/FluxML/Flux.jl)
* [MLDatasets]((https://juliaml.github.io/MLDatasets.jl/latest/))
 
To install these packages, run the following in the REPL:
 
```julia
Pkg.add("Flux")
Pkg.add("MLDatasets")
```
 
<br>
 
Load the packages we'll need:
 
```julia
using MLDatasets: MNIST
using Flux.Data: DataLoader
using Flux: onehotbatch
```
 
## Step1: Loading the MNIST data set
 
We load the MNIST train and test data from MLDatasets:
 
```julia-repl
julia> train_data = MNIST(:train)
dataset MNIST:
  metadata    =>    Dict{String, Any} with 3 entries
  split       =>    :train
  features    =>    28×28×60000 Array{Float32, 3}
  targets     =>    60000-element Vector{Int64}

julia> train_x, train_y = train_data[:];

julia> test_x, test_y = MNIST(:test)[:];
```
<br>
 
This code loads the MNIST train and test images as Float32 as well as their labels. The data set `train_x` is a 28×28×60000 multi-dimensional array. It contains 60000 elements and each one of it contains a 28x28 array. Each array represents a 28x28 image (in grayscale) of a handwritten digit. Moreover, each element of the 28x28 arrays is a pixel that represents the amount of light that it contains. On the other hand, `test_y` is a 60000 element vector and each element of this vector represents the label or actual value (0 to 9) of a handwritten digit.
 
## Step 2: Loading the dataset onto DataLoader
 
Before we load the data onto a DataLoader, we need to reshape it so that it has the correct shape for Flux. For this example, the MNIST train data must be of the same dimension as our model's input and output layers.
 
For example, if our model's input layer expects a 28x28x1 multi-dimensional array, we need to reshape the train and test data as follows:
 
```julia
train_x = reshape(train_x, 28, 28, 1, :)
test_x = reshape(test_x, 28, 28, 1, :)
```
<br>
 
Also, the MNIST labels must be encoded as a vector with the same dimension as the number of categories (unique handwritten digits) in the data set. To encode the labels, we use the [Flux's onehotbatch](https://fluxml.ai/Flux.jl/stable/data/onehot/#Batches-1) function:
 
```julia
train_y, test_y = onehotbatch(train_y, 0:9), onehotbatch(test_y, 0:9)
```
<br>
 
>**Note:** For more information on other encoding methods, see [Handling Data in Flux](https://fluxml.ai/Flux.jl/stable/data/onehot/).
 
Now, we load the train images and their labels onto a DataLoader object:
 
```julia
data_loader = DataLoader((train_x, train_y), batchsize=128, shuffle=true)
```
<br>
 
Notice that we set the DataLoader `batchsize` to 128. This will enable us to iterate over the data in batches of size 128. Also, by setting `shuffle=true` the DataLoader will shuffle the observations each time that iterations are re-started.
 
## Step 3: Iterating over the data
 
Finally, we can iterate over the 60000 MNIST train data in mini-batches (most of them of size 128) using the Dataloader that we created in the previous step. Each element of the DataLoader is a tuple `(x, y)`  in which `x` represents a 28x28x1 array and `y` a vector that encodes the corresponding label of the image.   
 
```julia
for (x, y) in data_loader
   @assert size(x) == (28, 28, 1, 128) || size(x) == (28, 28, 1, 96)
   @assert size(y) == (10, 128) || size(y) == (10, 96)
   ...
end
```
 
<br>
 
 
Now, we can create a model and train it using the `data_loader` we just created. For more information on building models in Flux, see [Model-Building Basics](https://fluxml.ai/Flux.jl/stable/models/basics/#Model-Building-Basics-1).
