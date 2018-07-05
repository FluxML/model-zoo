using Flux, Images, VideoIO

cd(@__DIR__)

isdir("data") ||
  run(`git clone https://github.com/americast/actions-dataset.git data --depth 1`)

const BATCH_SIZE = 3
const NUM_CLASSES = 3
const MAX_ITER = 20000

files(path) = joinpath.(path, readdir(path))
files(paths...) = vcat(files.(paths)...)

train = shuffle(files("data/walk", "data/jump", "data/run"))
test = shuffle(files("data/test"))

# Local response normalisation
function normalise(x; α = 10.0^(-4), β = 0.75, n = 5, k = 2)
    y = ones(size(x))
    for b in 1:size(x,4)
        for p in 1:size(x,1)
            for q in 1:size(x,2)
                for r in 1:size(x,3)
                    a_i = x[p,q,r,b].data[1]
                    j_min = max(1,r+1-div(n,2))
                    j_max = min(size(x,3),r+1+div(n,2))
                    a_j = 0
                    for j in j_min:j_max
                        a_j += (x[p,q,j,b]).data[1]^2
                    end
                    a_j *= α
                    a_j += k
                    a_j ^= β
                    y[p,q,r,b] = a_i/a_j
                end
            end
        end
    end
    return(param(y))
end

CNN() = Chain(
  Conv2D((11,11), 3=>96, relu, stride=3),
  normalise,
  x -> maxpool2d(x, 2),
  Conv2D((5,5), 96=>256, relu, pad =2),
  normalise,
  x -> maxpool2d(x, 2),
  Conv2D((3,3), 256=>384, relu, pad=2),
  normalise,
  x -> maxpool2d(x, 2),
  Conv2D((3,3), 384=>384, relu, pad=2),
  Conv2D((3,3), 384=>256, relu, pad=2),
  x -> maxpool2d(x, 2),
  x -> reshape(x, :, size(x,4)),
  Dense(4096, 4096, relu))

# Two similar models for context and fovea
m_c = CNN()
m_f = CNN()

# Single dense layer after concatenation of context and fovea
dense = Chain(
    Dense(8192,NUM_CLASSES + 1),
    softmax)

# Structure of the model
function inference(data_c, data_f)
    data_c = convert(Array{Float64,4},data_c)
    data_f = convert(Array{Float64,4},data_f)
    dense_c = m_c(data_c)
    dense_f = m_f(data_f)

    dense_net = vcat(dense_c,dense_f)
    dense_final = dense(dense_net)

    return dense_final

end

loss(x, y) = Flux.crossentropy(x, y)

# Evaluation done with batch size fixed to 1
function eval()
    loss_sum = 0
    counter = 0
    for file in test_files

        io = VideoIO.open(file)
        f = VideoIO.openvideo(io)
        img = read(f)

        while !eof(f)
            read!(f, img)
            img_context = imresize(img,(89,89))
            img_context = convert(Array{Float64,3},rawview(ChannelView(img_context)))
            img_context = reshape(img_context,(89,89,3,1))

            img_fovea = img[div(size(img,1),2)-44:div(size(img,1),2)+44,div(size(img,2),2)-44:div(size(img,2),2)+44]
            img_fovea = convert(Array{Float64,3},rawview(ChannelView(img_fovea)))
            img_fovea = reshape(img_fovea,(89,89,3,1))

            ImageView.closeall()
            ImageView.imshow(img)

            action_type = split(split(file,"/")[3],".")[1]
            label = 4
            if (action_type=="walk")
                label = 1
            elseif (action_type=="run")
                label = 2
            elseif (action_type=="jump")
                label = 3
            end

            data = inference(img_fovea, img_context)

            result = Flux.onehot(maximum(data),data)

            print("Prediction: ")
            if (result[1])
                println("walk")
            elseif (result[2])
                println("run")
            elseif (result[3])
                println("jump")
            else
                println("Unknown")
            end

            hot = zeros(NUM_CLASSES + 1)
            hot[label] = 1

            loss_sum+=loss(data,hot)
            counter+=1
        end
    end
    loss_sum/=counter
    println("Loss: ", loss_sum.data)
    ImageView.closeall()
end


opt = ADAM(params(inference))

for iter in 1:MAX_ITER
    println("Starting iteration $iter")
    for file in files

        io = VideoIO.open(file)
        f = VideoIO.openvideo(io)
        img = read(f)

        count=0
        data_context = []
        data_fovea = []
        labels = []


        while !eof(f)
            if (count == BATCH_SIZE)
                count = 0
                data_context = reshape(data_context,(89,89,3,BATCH_SIZE))
                data_fovea = reshape(data_fovea,(89,89,3,BATCH_SIZE))

                final_labels = []

                for i in 1:length(labels)
                    label = labels[i]
                    hot = zeros(NUM_CLASSES + 1)
                    hot[label] = 1
                    append!(final_labels, hot)
                end

                final_labels = reshape(final_labels,(NUM_CLASSES + 1, BATCH_SIZE))

                data = [(inference(data_context, data_fovea),final_labels)]

                Flux.train!(loss, data, opt, cb = () -> println("training"))

                data_context = []
                data_fovea = []
                labels = []

            end

            read!(f, img)
            img_context = imresize(img,(89,89))
            img_context = convert(Array{Float64,3},rawview(ChannelView(img_context)))
            img_context = reshape(img_context,(89,89,3))
            append!(data_context,img_context)

            img_fovea = img[div(size(img,1),2)-44:div(size(img,1),2)+44,div(size(img,2),2)-44:div(size(img,2),2)+44]
            img_fovea = convert(Array{Float64,3},rawview(ChannelView(img_fovea)))
            img_fovea = reshape(img_fovea,(89,89,3))
            append!(data_fovea,img_fovea)

            action_type = split(file,"/")[2]
            if (action_type=="walk")
                push!(labels,1)
            elseif (action_type=="run")
                push!(labels,2)
            elseif (action_type=="jump")
                push!(labels,3)
            else
                push!(labels,4)
            end
            count+=1
        end
    end
    eval()
    println("Iteration $iter complete")
end
