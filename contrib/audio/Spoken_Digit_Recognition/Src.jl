using DSP,WAV
using PyCall
using PyPlot
using MFCC
using FFTW
using Flux
using Printf,BSON
using Flux: onehotbatch, onecold, crossentropy, throttle, Conv,relu
using Base.Iterators: partition
using StatsBase
using MLLabelUtils,MLDataPattern
IpY = pyimport("IPython")
using Images

#Loading the data
cd("./@__dir__")     #Replace the dir with your directory where you have unzipped the Audio Data
A = readdir("./Spoken_Digit/recordings")
cd("./Spoken_Digit/recordings")
X = []
X_fs = []
Y = []
for i in 1:length(A)
    s,fs = wavread(A[i])
    push!(X,s)
    push!(X_fs,fs)
    push!(Y,Int(A[i][1]-'0'))
end
cd("./../../")

#Converting the audio data into Spectrogram images which will then be used for training the model 
imgs = []
for i in 1:length(X)
    b = spectrogram(X[i][:,1])
    push!(imgs,b.power)
end

labels = Y;
#Shuffle the data before minibatch formation 
imgs_,labels_ = shuffleobs((imgs,labels));

#Normalising the data
for i in 1:length(imgs)
    imgs[i] = Flux.normalise(imgs[i],dims=2)
end

#Use 85% of the total data as train data and rest as test data
train_X,train_Y = img_[1:1701],labels_[1:1701]

#Since the Spectrogram images of different audio signals will also be different, so they are converted to a common size
img_size = (256,32)
m,n = img_size

#Function for minibatch formation
function make_minibatch(X,Y,idxs)
    X_batch = Array{Float32}(undef,(img_size)..., 1, length(idxs)) #Declaring an array of images as a batch 
    for i in 1:length(idxs)
        img = Float32.(imresize((X[idxs[i]]),(img_size)...))#Resize the image
        X_batch[:, :, :, i] = img
    end
    Y_batch = onehotbatch(Y[idxs], 0:9) #Onehot encode the labels
    return (X_batch, Y_batch)
end


#Dividing the data into minibatches
mb_indices = [] #Array of indices to be loaded as minibatches
batch_size = 32
for i in range(1,length(train_Y)-1,step = batch_size)
    idxs = []
    for j in i:i+batch_size-1
        push!(idxs,j)
    end
    push!(mb_indices,idxs)
end
train_set = [make_minibatch(train_X,train_Y,mb_indices[i]) for i in 1:(size(mb_indices)[1]-1)];

#Test data as a single batch
batch_size=300
ind = []
for i in 1701:2000
    push!(ind,i)
end
test_set = [make_minibatch(imgs_,labels_,ind)];

@info("Constructing model...")
model = Chain(
    # First convolution, operating upon a m*n image
    Conv((3, 3), 1=>64, pad=(1,1), relu),
    MaxPool((2,2)),
    BatchNorm(64,relu),

    # Second convolution, operating upon a m/2*n/2 image
    Conv((3, 3), 64=>32, pad=(1,1), relu),
    MaxPool((2,2)),
    BatchNorm(32,relu),
    Dropout(0.10),
    
    # Reshape 3d tensor into a 2d one, at this point it should be (m/4,n/4,32, N)
    # which is where we get the 2048 in the `Dense` layer below:
    x -> reshape(x, :, size(x, 4)),
    Dense(Int(floor(m/4)*floor(n/4)*32), 128,relu),
    
    Dense(128,10),

    # Finally, softmax to get nice probabilities
    softmax,
)

function loss(x, y)
    # We augment `x` a little bit here, adding in random noise
    x_aug = x .+ 0.1f0*(randn(eltype(x), size(x)))

    y_hat = model(x_aug)
    return crossentropy(y_hat, y)
end
#Accuracy function
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

#Training the data
opt = ADAM(0.001)
epochs = 15

@info("Beginning training loop...")
best_acc = 0.0
last_improvement = 0
for epoch_idx in 1:epochs
    global best_acc, last_improvement
    # Train for a single epoch
    Flux.train!(loss, params(model), train_set, opt)
    x,y = train_set[1] 
    print("Epoch[$epoch_idx]: Train_Loss: ",loss(x,y),"\n")
    
    # Calculate accuracy:
    acc = accuracy(test_set[1]...)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))

    # If our accuracy is good enough, quit out.
    if acc >= 0.95
        @info(" -> Early-exiting: We reached our target accuracy of 95.0%")
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if acc >= best_acc
        @info(" -> New best accuracy! Saving model out to MNIST_Speech.bson") #Here, model is saved as MNIST_Speech.bson        
        BSON.@save joinpath(dirname(@__FILE__), "./MNIST_Speech.bson") model epoch_idx acc
        best_acc = acc
        last_improvement = epoch_idx
    end

    # If we haven't seen improvement in 5 epochs, drop our learning rate:
    if epoch_idx - last_improvement >= 5 && opt.eta > 1e-4
        opt.eta /= 10.0
        @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

        # After dropping learning rate, give it a few epochs to improve
        last_improvement = epoch_idx
    end

    if epoch_idx - last_improvement >= 10
        @warn(" -> We're calling this converged.")
        break
    end
end


