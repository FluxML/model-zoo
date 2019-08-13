# BatchNorm Wrapper Utilities
expand_dims(x,n::Int) = reshape(x,ones(Int64,n)...,size(x)...)
function squeeze(x) 
    if size(x)[end] != 1
        return dropdims(x, dims = tuple(findall(size(x) .== 1)...))
    else
        # For the case BATCH_SIZE = 1
        int_val = dropdims(x, dims = tuple(findall(size(x) .== 1)...))
        return reshape(int_val,size(int_val)...,1)
    end
end

# BatchNorm Wrapper
function BatchNormWrap(out_ch)
    Chain(x->expand_dims(x,2),
    BatchNorm(out_ch),
    x->squeeze(x))
end

flatten(x) = reshape(x,prod(size(x)[1:end-1]),size(x)[end])

function norm(x)
    convert(CuArray{Float32},2.0 .* x .- 1.0)
end

function denorm(x)
   convert(CuArray{Float32},((x .+ 1.0)./(2.0) ))
end

# Image loading utilities #
function load_image(filename)
    img = load(filename)
    img = Float32.(channelview(img))
end

function load_dataset(HR_path,LR_path)
    img_name_HR = []
    img_name_LR = []
    for r in readdir(HR_path)
        img_HR_path = string(HR_path,r)
        img_LR_path = string(LR_path,string(r[1:end-4],"x4",".png"))
        push!(img_name_HR,img_HR_path)
        push!(img_name_LR,img_LR_path)
    end
    return img_name_HR,img_name_LR
end

function get_batch(img_name_HR,img_name_LR,H::Int,W::Int)
   imgsHR = []
   imgsLR = []
   for (i,name_HR) in enumerate(img_name_HR)
        push!(imgsHR,load_image(name_HR))
        push!(imgsLR,load_image(img_name_LR[i]))
   end
   return reshape(cat(imgsHR...,dims=4),W,H,3,length(imgsHR)),reshape(cat(imgsLR...,dims=4),div(W,4),div(H,4),3,length(imgsLR))
end

function bce(ŷ, y)
    -y.*log.(ŷ .+ 1f-10) - (1  .- y).*log.(1 .- ŷ .+ 1f-10)
end

function logitbinarycrossentropy(logŷ, y)
  mean((1 .- y).*logŷ .- logσ.(logŷ))
end
