using Images, Augmentor

function generate_crops(img_path, num)
    img = load(img_path)
    save_path = img_path[1:end-4]
    dim_x = size(img, 1)
    dim_y = size(img, 2)
    if(dim_x >= 225 && dim_y >= 225)
        if(num == 0)
            choice = rand(0:4)
            if(choice==0)
                save(img_path, augment(img, CropSize(224, 224)))
            elseif(choice==1)
                save(img_path, augment(img, Crop(1:224, 1:224)))
            elseif(choice==2)
                save(img_path, augment(img, Crop(1:224, (dim_y - 223):dim_y)))
            elseif(choice==3)
                save(img_path, augment(img, Crop((dim_x - 223):dim_x, 1:224)))
            else
                save(img_path, augment(img, Crop((dim_x - 223):dim_x, (dim_y - 223):dim_y)))
            end
        else
            save(save_path * "_1.jpg", augment(img, CropSize(224, 224)))
            save(save_path * "_2.jpg", augment(img, Crop(1:224, 1:224)))
            save(save_path * "_3.jpg", augment(img, Crop(1:224, (dim_y - 223):dim_y)))
            save(save_path * "_4.jpg", augment(img, Crop((dim_x - 223):dim_x, 1:224)))
            save(save_path * "_5.jpg", augment(img, Crop((dim_x - 223):dim_x, (dim_y - 223):dim_y)))
        end
    end
    if(num==1)
        rm(img_path)
    end
end

train_dir = "./places365_standard/train"
val_dir = "./places365_standard/val"

train_img_dirs = [joinpath(train_dir, i) for i in readdir(train_dir)]
val_img_dirs = [joinpath(val_dir, i) for i in readdir(val_dir)]

for i in train_img_dirs
    for j in [joinpath(i, p) for p in readdir(i)]
        generate_crops(j, 0)
    end
    info("Processing of 1 category done")
end

info("Training Images Processed")

for i in val_img_dirs
    for j in [joinpath(i, p) for p in readdir(i)]
        generate_crops(j, 1)
    end
    info("Processing of 1 category done")
end

info("Validation Images Processed")

