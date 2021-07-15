imload(x) = float.(permutedims(channelview(load(x)), [3,2,1]))

train_dir = "./Scenes Dataset/Train_Data"

val_dir = "./Scenes Dataset/Validation_Data"

labels = [*((all(isnumber, i[end])?i[1:end-1]:i)...) for i in split.(readdir(train_dir), "-")]

train_data = []
train_labels = []

val_data = []
val_labels = []

label_count = 0

for i in readdir(train_dir)
    info("Reading Training Data from $i")
    for j in readdir(joinpath(train_dir, i))
        push!(train_data, imload(joinpath(train_dir, i, j)))
        push!(train_labels, label_count)
    end
    label_count += 1
end

info("Starting to shuffle the training data")

shufflelist = randperm(length(train_data))
train_data = train_data[shufflelist]
train_labels = train_labels[shufflelist]

info("Shuffling dataset complete")

label_count = 0

for i in readdir(val_dir)
    info("Reading Validation Data from $i")
    for j in readdir(joinpath(val_dir, i))
        push!(val_data, imload(joinpath(val_dir, i, j)))
        push!(val_labels, label_count)
    end
    label_count += 1
end

info("Loading Training and Validation Data Complete")

info("Breaking the data into batches")

train_labels = onehotbatch(train_labels, 0:(label_count-1))
train = [(cat(4, (train_data[i])...), train_labels[:,i]) for i in partition(1:length(train_data), 16)]
val_labels = onehotbatch(val_labels, 0:(label_count-1))
val_data = cat(4, val_data...)

info("Data Partition Complete")
