using Images
using Base.Iterators.partition

include("googlenet.jl")

function read_from_path(img_path)
    z = load(img_path)
    z2 = channelview(z)
    if(ndims(z2) == 2)
        z2 = channelview(RGB.(z))
    end
    float.(permutedims(z2, [3,2,1]))
end

train_dir = "./train"

item_mapping = Dict()
count = 1

for i in readdir(train_dir)
    item_mapping[i] = count
    count += 1
end

f = open("train.txt", "r")

train_imgs_paths = []
train_labels = []

while(!eof(f))
    push!(train_imgs_paths, readline(f))
    push!(train_labels, item_mapping[split(train_imgs_paths[end], "/")[end-1]])
end

inds = randperm(length(train_imgs_paths))

train_imgs_paths = train_imgs_paths[inds]
train_labels = train_labels[inds]

loss(x, y) = Flux.crossentropy(x, y)

accuracy(x, y) = mean(argmax(x, 1:365) .== argmax(y, 1:365))

nepochs = 1

a = time()
c = 0

@epochs nepochs begin
    for i in partition(1:length(train_labels), 10000)
        train_imgs = [read_from_path(j) for j in train_imgs_paths[i]]
        train_labs = Flux.onehotbatch(train_labels[i], 1:365)
        train = [(cat(4, train_imgs[j]...), train_labs[:,j]) for j in partition(1:length(train_imgs), 16)]
        for data in train
            forward = (model(data[1] |> gpu), data[2] |> gpu)
            l = loss(forward[1], forward[2])
            info("Loss = $l")
            info("Accuracy = $(accuracy(forward[1], forward[2]))")
            Flux.back!(l)
            opt()
            if(time()-a>1800)
                c += 1
                @save "model_checkpoint_$c.bson" model
                a = time()
            end
        end
        info("Training on 1 partition done")
    end
end
