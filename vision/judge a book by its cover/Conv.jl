using Flux
using Flux: @epochs, onehotbatch, crossentropy, throttle
using Images
using FileIO
using CSV
using Statistics

train_path = "G:\\Book-Cover-Train\\"
train_csv = "F:\\book-dataset\\Task1\\book30-listing-train.csv"

test_path = "G:\\Book-Cover-Test\\"
test_csv = "F:\\book-dataset\\Task1\\book30-listing-test.csv"

train_imglist = readdir(train_path)
test_imglist = readdir(test_path)

train_setsize = length(train_imglist)
test_setsize = length(test_imglist)

batch_size = 1000


function create_dataset(indexs; path, csv, images)
    dataset = CSV.read(csv)
    X = Array{Float32}(undef, 100, 100, 3, length(indexs))
    for i = 1:length(indexs)
        img = load(string(path, images[i]))
        img = channelview(imresize(img, 100, 100))
        img = Float32.(permutedims(img, (2, 3, 1)))
        X[:, :, :, i] = img
    end
    Y = onehotbatch(dataset[indexs[1]:indexs[end], 6], 0:29)
    return (X, Y)
end

indexs = Base.Iterators.partition(1:train_setsize, batch_size)
train_set = [create_dataset(
    i;
    path = train_path,
    csv = train_csv,
    images = train_imglist,
) for i in indexs]

test_set = create_dataset(
    1:test_setsize;
    path = test_path,
    csv = test_csv,
    images = test_imglist,
)

m = Chain(
    Conv((3, 3), 3 => 32, pad = (1, 1), relu),
    MaxPool((2, 2)),

    Conv((3, 3), 32 => 64, pad = (1, 1), relu),
    MaxPool((2, 2)),

    Conv((3, 3), 64 => 256, pad = (1, 1), relu),
    MaxPool((2, 2)),

    Conv((2, 2), 256 => 512, pad = (1, 1), relu),
    MaxPool((2, 2)),

    x -> reshape(x, :, size(x, 4)),
    Dense(18432, 256, relu),
    Dense(256, 30),
    softmax,
)

loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) = mean(Flux.onecold(m(x)) .== Flux.onecold(y))

function cbfunc()
    ca = accuracy(test_set...)
    @show(ca)
    cl = loss(test_set...)
    @show(cl)
end

opt = ADAM()
@epochs 5 Flux.train!(loss, params(m), train_set, opt, cb = throttle(cbfunc, 3))
