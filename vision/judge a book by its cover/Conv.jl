# After a lot of debugging and asking around
# The fix was found to be to remove every conv layer.
# Previously, whilst using the conv model, the accuracy was stuck at exactly 0.8 and loss at exactly 3.33253
# Changing the model to be that of a simple dense one seems to bring the accuracy up to what is expected.
# In the research paper, authors were able to achive a general accuracy of 24% in the model which they trained from the ground up
# In doing so they used the whole 30000 image dataset. This model uses 1/3 of that datset and around 500 testing images because of processing constraints
# The accuracy achived after 20 epochs was right at ~12%. Considering that it uses a third of the datset, this is pretty decent.

using Flux
using Flux: onehotbatch, crossentropy, throttle
using Images
using FileIO
using CSV
using Statistics

const train_path = "G:\\mlimgs\\Book-Train-FULL\\" # contains the training images
const train_csv = "F:\\book-dataset\\Task1\\book30-listing-train1.csv" # This file contains the labels(genre) for the training images

const test_path = "G:\\mlimgs\\Book-Test-FULL\\" #Contains the testing dataset
const test_csv = "F:\\book-dataset\\Task1\\book30-listing-test1.csv" # This file contains the labels(genre) for the testing images

const train_dataset  = CSV.read(train_csv)  # read the csv for training labels
                                            # The CSVs have 2 colums: first of genre and second of name of the book
                                            # The name of the book isn't required for the functioning of the model but is included for debugging purposes

const test_dataset  = CSV.read(test_csv) # read the csv for testing labels

# find the total number of images in sets so we can correctly divide the dataset into batches
const train_imglist = readdir(train_path)
const test_imglist = readdir(test_path)
const train_setsize = length(train_imglist)
const test_setsize = length(test_imglist)

# Self-explainatory Hyper Parameters
const batch_size = 400
const imsize = 60
const epochs = 20
const learning_rate = 0.0001

function create_batch(indexs; path, csv, dataset)
    X = Array{Float32}(undef, imsize*imsize*3, length(indexs))  # everytime this function is called a new batch is created with the correct size
                                                                # It should be able to hold multiple flattened images (flattened because we're using a dense network)
                                                                # Thats why it is shaped like (size_of_image,no._of_images)
    for (p,i) in enumerate(indexs)
        img = load(string(path,i,".png")) # The images are labeled like 1.png, 2.png, and so on.
        img = channelview(RGB.(imresize(img, imsize, imsize)))
        img = reshape(Float32.(img),(imsize*imsize*3))  # The current image has 3 layers of 60 by 60 pixels all compiled into a 3D array
                                                        # We need the image in a flat array so we reshape into a flat array for it to be eligible to be added to array X
        X[:, p] = img # add the img to X.

    end
    Y = onehotbatch(dataset[indexs, 1], 0:29)
    return (X, Y)
end

const indexs = Base.Iterators.partition(1:train_setsize, batch_size)

const test_set = create_batch(
    1:test_setsize;
    path = test_path,
    csv = test_csv,
    dataset = test_dataset
)

@info "creating the model"
# I've tried using a conv net described in the paper but that
# yields an accuracy of 0.8 and has a lot of inconsistencies  with it
# People over at #julia-bridged (after a LONG thread of conversation) told me
# to just change out the conv with a dense model. Sure enough, it started to behave like its supposed to

m = Chain(
    Dense(imsize*imsize*3, 512, relu), # we're expecting an image array
    Dense(512, 64),
    Dense(64, 30),
    softmax,
)

loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) = mean(Flux.onecold(m(x)) .== Flux.onecold(y))

opt = ADAM(learning_rate)

function cbfunc()
    ca = accuracy(test_set...)
    print("batch_acc: ",string(ca),"; ")
    cl = loss(test_set...)
    println("batch_loss: ",string(cl))

end

for e in 1:epochs
    @info "Epoch no.-> $e"
    b = 1
    for i in indexs
        println("Batch no. -> $b")
        train_batch = [create_batch(i; path = train_path, csv = train_csv, dataset = train_dataset)] # we load every batch before training
                                                                                                     # This way we dont have to load the whole big dataset into one array
        Flux.train!(loss, params(m), train_batch , opt, cb = cbfunc)
        b+=1
    end

end

println("Final acc and loss : ")
cbfunc()
