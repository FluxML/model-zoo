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
  
  # Convert the text based names to 0 or 1
  labels = Flux.onehotbatch(labels, [0,1])
  
  # Load all of the images
  imgs = Images.load.(imgs_paths)
  
  # Re-size the images based on imagesize from above (most models use 224 x 224)
  imgs = map(img -> Images.imresize(img, nsize...), imgs)
  
  # Change the dimensions of each image, switch to gray scale
  imgs = map(img -> permutedims(channelview(img), (3,2,1)), imgs)
  # Result is two 3D arrays representing each image.
  
  # Concatenate the two images into a single 4D array
  imgs = cat(imgs..., dims = 4)
  
  # Convert the images to float form and return them along with the labels
  Float32.(imgs), labels
end
