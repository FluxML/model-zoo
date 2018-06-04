using Images
using Base.Iterators.partition

include("inceptionv3.jl")

read_from_path(img_path) = float.(channelview(load(img_path)))

train_dir = "./train"

item_mapping = Dict()
count = 1

for i in readdir(train_dir)
    item_mapping[i] = count
    count += 1
end

f = open(joinpath(train_dir, "train.txt"), "r")

train_imgs_paths = []
train_labels = []

while(!eof(f))
    push!(train_imgs_paths, readline(f))
    push!(train_labels, item_mapping[split(train_imgs[end], "/")[end-1]])
end

inds = randperm(length(train_imgs_paths))

train_imgs_paths = train_imgs_paths[inds]
train_labels = train_labels[inds]

loss(x, y) = Flux.crossentropy(x, y)

accuracy(x, y) = mean(argmax(x, 1:365) .== argmax(y, 1:365))

epochs = 1

a = time()
c = 0

@epochs begin
    for i in partition(1:length(train_labels), 10000)
        train_imgs = [read_from_path(j) for j in train_imgs_paths[i]]
        train_labs = Flux.onehotbatch(train_labels[i], 1:365)
        train = [(cat(4, train_imgs[j]), train_labs[:,j]) for j in partition(1:length(train_imgs), 128)]
        for data in train
            forward = (model(data[1] |> gpu), data[2] |> gpu)
            l = loss(forward...)
            info("Loss = $l")
            info("Accuracy = $(accuracy(forward...))")
            Flux.back!(l)
            opt()
            if(a-time()>1800)
                c += 1
                @save "model_checkpoint_$c.bson" model
                a = time()
            end
        end
        info("Training on 1 partition done")
    end
end
