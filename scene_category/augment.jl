const data_dir = joinpath(@__DIR__, "Scenes Dataset")

const data_files = [joinpath(data_dir, i) for i in readdir(data_dir)]

#---------------------------Crop the Images into 224x224---------------------------

function generate_crops(img_path)
    # Since we are using models originally developed for Imagenet we are going to generate 5 crops of dimensions 224 x 224
    img = load(img_path)
    save_path = img_path[1:end-4]
    dim_x = size(img, 1)
    dim_y = size(img, 2)
    if(dim_x >= 225 && dim_y >= 225)
        save(save_path * "_1.jpg", augment(img, CropSize(224, 224)))
        save(save_path * "_2.jpg", augment(img, Crop(1:224, 1:224)))
        save(save_path * "_3.jpg", augment(img, Crop(1:224, (dim_y - 223):dim_y)))
        save(save_path * "_4.jpg", augment(img, Crop((dim_x - 223):dim_x, 1:224)))
        save(save_path * "_5.jpg", augment(img, Crop((dim_x - 223):dim_x, (dim_y - 223):dim_y)))
    end
    rm(img_path)
end

for i in 1:length(data_files)
    data_files_in = [joinpath(data_files[i], j) for j in readdir(data_files[i])]
    for j in 1:length(data_files_in)
        images = [joinpath(data_files_in[j], k) for k in readdir(data_files_in[j])]
        for k in images
            generate_crops(k)
        end
        info("Category $(j) is processed")
    end
    info("Processing Directory $(i) Completed")
end

info("Generation of Crops Complete")

#---------------------------Augment the Cropped Images-----------------------------

const aug_data_dir = ["./Scenes\ Dataset/16-scenes",
                  "./Scenes\ Dataset/1-novel",
                  "./Scenes\ Dataset/4-scenes"]

# After this operation is performed we get 50 1-novel, 160 4-scenes, 320 16-scenes, 320 64-scenes and 340 68-scenes
# Overall we have 160 categories to predict

function augment_images(img_path, set)
    img = load(img_path)
    save_path = img_path[1:end-4]
    save(save_path * "_1.jpg", augment(img, FlipX()))
    save(save_path * "_2.jpg", augment(img, Rotate(-5:5)))
    save(save_path * "_3.jpg", augment(img, Rotate(-5:5)))
    if(set!=16)
        save(save_path * "_4.jpg", augment(img, Rotate(-5:5)))
        save(save_path * "_5.jpg", augment(img, Rotate(-5:5)))
        save(save_path * "_6.jpg", augment(img, ElasticDistortion(7, scale = 0.1)))
        save(save_path * "_7.jpg", augment(img, ElasticDistortion(10, scale = 0.1)))
    end
    if(set==1)
        save(save_path * "_8.jpg", imfilter(img, Kernel.gaussian(3)))
        save(save_path * "_9.jpg", imfilter(img, Kernel.gaussian(1)))
    end
end

for (q, p) in enumerate([16,1,4])
    local data_files = [joinpath(aug_data_dir[q], pq) for pq in readdir(aug_data_dir[q])]
    for i in 1:length(data_files)
        images = [joinpath(data_files[i], j) for j in readdir(data_files[i])]
        for j in 1:length(images)
            augment_images(images[j], p)
        end
        info("Augmenting $i images done")
    end
end

info("Augmentation of Images Complete")

#-------------------Resize the Images back to Proper Dimensions--------------------

# Some images might have changed dimensions due to Augmentation
function resize_images(img_path)
    img = load(img_path)
    if(size(img)!=(224,224))
        save(img_path, augment(img, CropSize(224,224)))
    end
end

for q in 1:5
    local data_files_in = [joinpath(data_files[q], pq) for pq in readdir(data_files[q])]
    for i in 1:length(data_files_in)
        images = [joinpath(data_files_in[i], j) for j in readdir(data_files_in[i])]
        for j in 1:length(images)
            resize_images(images[j])
        end
        info("Resizing $i images done")
    end
end

#-----------------------Creating Train Test Split--------------------------

# Since the dataset is too small we split the data into only validation and train sets
validation_data = joinpath(data_dir, "Validation_Data")
train_data = joinpath(data_dir, "Train_Data")
mkdir(validation_data)
mkdir(train_data)

function make_split(images, path)
    total_images = length(images)
    # Take 10% of the images as Validation Data
    split = Int64(ceil(0.1 * total_images))
    indices = randperm(total_images)
    split_indices = indices[1:split]
    train_indices = indices[(split+1):end]
    val_dir_name = joinpath(validation_data, splitdir(path)[2])
    train_dir_name = joinpath(train_data, splitdir(path)[2])
    mkdir(val_dir_name)
    mkdir(train_dir_name)
    for i in split_indices
        save_path = joinpath(val_dir_name, splitdir(images[i])[end])
        mv(images[i],save_path)
    end
    for i in train_indices
        save_path = joinpath(train_dir_name, splitdir(images[i])[end])
        mv(images[i],save_path)
    end
end

for q in 1:5
    data_files_in = [joinpath(data_files[q], pq) for pq in readdir(data_files[q])]
    for i in 1:length(data_files_in)
        images = [joinpath(data_files_in[i], j) for j in readdir(data_files_in[i])]
        make_split(images, data_files_in[i])
        info("Making Validation Splits of $i images done")
    end
end

