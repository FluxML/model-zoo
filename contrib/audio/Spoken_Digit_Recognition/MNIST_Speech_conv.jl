#Importing the required Packages
using DSP,WAV
using PyCall
using Plots,PyPlot
using PaddedViews
using MFCC
using FFTW
using MLLabelUtils
using Flux
using Printf,BSON
using Flux: onehotbatch, onecold, crossentropy, throttle, Conv,relu
using Base.Iterators: repeated, partition
using StatsBase
using MLLabelUtils,MLDataPattern
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators:partition
using Images


cd("@__dir__")

A = readdir("./Spoken_Digit/recordings")

cd("./Spoken_Digit/recordings")
#Loading the WAV Audio files
X = []
X_fs = []
Y = []
for i in 1:length(A)
    s,fs = wavread(A[i])
    push!(X,s)
    push!(X_fs,fs)
    push!(Y,Int(A[i][1]-'0')) #First letter of the dataset gives the Digit labelling
end
cd("../..")

#To display Audio File
#= ########
IpY = pyimport("IPython")
IpY.display.Audio(A[453])
PyPlot.plot(X[1])
=# ########

#Padding each audio file to length of 10000
for i in 1:length(X)
    X[i] = Array(PaddedView(0,X[i],(10000,1)))
end


#Function to find periodogram of the first audio file
Y1 = periodogram(X[1][:,1])

#Function to plot the Periodogram freq vs Power
PyPlot.plot(Y1.freq, DSP.pow2db.(Y1.power))

#Function to Plot Spectrogram
c = PyPlot.specgram(X[1][:,1],Fs = X_fs[1]) #Spectrogram of Padded audio data
a,f = wavread(A[1])
PyPlot.specgram(a[:,1],Fs = f) #Specgram of Unpadded audio data

#Loading Specgram Data of each digit as the basis to apply Neural Network model on!
imgs = []
for i in 1:length(X)
    b = PyPlot.specgram(X[i][:,1],Fs = X_fs[i])
    push!(imgs,b[1])
end

labels = Y;

#Shuffling the Data
imgs_,labels_ = shuffleobs((imgs,labels));

#Normalising the imgs data
for i in 1:length(imgs)
    imgs[i] = Flux.normalise(imgs[i],dims=2)
end

#Splitting the data to use 15% as test
(train_X, train_Y), (test_X, test_Y) = splitobs((imgs_, labels_); at = 0.85);

#Function to Make minibatches of the Training Data
function make_minibatch(X,Y,idxs)
    X_batch = Array{Float32}(undef,(img_size)..., 1, length(idxs))
    for i in 1:length(idxs)
        img = Float32.(imresize((X[idxs[i]]),(img_size)...)) #resize the Specgram data to a fixed size
        X_batch[:, :, :, i] = img
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end

mb_indices = [] 
batch_size = 32
img_size = (128,64)
#Generating the indices of the minibatches to load on as Training Data
for i in range(1,length(train_Y)-1,step = batch_size)
    idxs = []
    for j in i:i+batch_size-1
        push!(idxs,j)
    end
    push!(mb_indices,idxs)
end
#train_set gives an array of minibatches each of size 32
train_set = [make_minibatch(train_X,train_Y,mb_indices[i]) for i in 1:(size(mb_indices)[1]-1)]

@info("Constructing model...")
model = Chain(
    # First convolution, operating upon a 128*64 image
    Conv((3, 3), 1=>64, pad=(1,1), relu),
    MaxPool((2,2)),
    BatchNorm(64,relu),

    # Second convolution, operating upon a 64*32 image
    Conv((3, 3), 64=>32, pad=(1,1), relu),
    MaxPool((2,2)),
    BatchNorm(32,relu),
    Dropout(0.15),
    
    # Reshape 3d tensor into a 2d one, at this point it should be (32,16,32, N)
    # which is where we get the 2048 in the `Dense` layer below:
    x -> reshape(x, :, size(x, 4)),
    Dense(32*16*32, 128,relu),
    
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
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

#15% of total dataset = 300 Audio files are used as Test Data
batch_size=300
img_size = (128,64)
ind = []
for i in 1701:2000
    push!(ind,i)
end
test_set = [make_minibatch(imgs_,labels_,ind)]; #get an array of single Minibatch of size 300 to test the accuracy of the Model

opt = ADAM(0.001)

@info("Beginning training loop...")
best_acc = 0.0
last_improvement = 0
for epoch_idx in 1:20
    global best_acc, last_improvement
    # Train for a single epoch
    Flux.train!(loss, params(model), train_set, opt)
    x,y = train_set[1] 
print(loss(x,y))
    
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
        @info(" -> New best accuracy! Saving model out to Digit_Speech.bson")
        BSON.@save joinpath(dirname(@__FILE__), "Digit_Speech.bson") model epoch_idx acc #Saving the model as Digit_Speech.bson file
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

#Loading the model
BSON.@load "Digit_Speech.bson" model

print(accuracy(test_set[1]...))


