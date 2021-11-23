using Flux, Images
using StatsBase: sample, shuffle

const PATH = joinpath(@__DIR__, "train")
const FILES = joinpath.(PATH, readdir(PATH))
if isempty(readdir(PATH))
  error("Empty train folder - perhaps you need to download and extract the kaggle dataset.")
end

# Get all of the files with "dog" in the name
const DOGS = filter(x -> occursin("dog", x), FILES)

# Get all of the files with "cat" in the name
const CATS = filter(x -> occursin("cat", x), FILES)

# Takes in the number of requested images per batch ("n") and image size
# Returns a 4D array with images and an array of labels
function load_batch(n = 10, nsize = (224,224); path = PATH)
  if ((batchsize % 2) != 0)
      print("Batch size must be an even number")
  end
  # Sample N dog images and N cat images, shuffle, and then combine them into a batch
  imgs_paths = shuffle(vcat(sample(DOGS, Int(n/2)), sample(CATS, Int(n/2))))
  
  # Generate the image label based on the file name
  labels = map(x -> occursin("dog.",x) ? 1 : 0, imgs_paths)
  # Here, dog is set to 1 and cat to 0
  
  # Convert the text based names to 0 or 1 (one hot encoding)
  labels = Flux.onehotbatch(labels, [0,1])
  
  # Load all of the images
  imgs = Images.load.(imgs_paths)
  
  # Re-size the images based on imagesize from above (most models use 224 x 224)
  imgs = map(img -> Images.imresize(img, nsize...), imgs)
  
  # Change the dimensions of each image, switch to gray scale. Channel view switches to...
  # a 3 channel 3rd dimension and then (3,2,1) makes those into seperate arrays.
  # So we end up with [:, :, 1] being the Red values, [:, :, 2] being the Green values, etc
  imgs = map(img -> permutedims(channelview(img), (3,2,1)), imgs)
  # Result is two 3D arrays representing each image
  
  # Concatenate the two images into a single 4D array and add another extra dim at the end
  # which shows how many images there are per set, in this case, it's 2
  imgs = cat(imgs..., dims = 4)
  # This is requires since the model's input is a 4D array
  
  # Convert the images to float form and return them along with the labels
  # The default is float64 but float32 is commonly used which is why we use it
  Float32.(imgs), labels
end
