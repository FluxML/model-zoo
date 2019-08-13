function random_crop(imgA,imgB,scale=256)
    W,H = size(imgA) # We have W = H
    diff = W - scale
    i = rand(1:diff)
    
    return imgA[i:i+scale-1,i:i+scale-1],imgB[i:i+scale-1,i:i+scale-1]
end

function random_jitter(img;RESIZE_SCALE = 286)
    A,B = img[:,1:256],img[:,257:end]
    
    A = imresize(A,(RESIZE_SCALE,RESIZE_SCALE))
    B = imresize(B,(RESIZE_SCALE,RESIZE_SCALE))
    
    A,B = random_crop(A,B)
    return cat(A,B,dims=2)
end

function load_image(filename)
    img = load(filename)
    # img = random_jitter(img)
    img = Float32.(channelview(img))
end

function load_dataset(path,imsize)
    imgs = []
    for r in readdir(path)
        img_path = string(path,r)
        push!(imgs,img_path)
    end
    imgs
end

function get_batch(files,imsize)
   """
   files : array of image names in a path
   """
   imgsA = []
   imgsB = []
   for file in files
        push!(imgsA,load_image(file)[:,:,1:256])
        push!(imgsB,load_image(file)[:,:,257:end])
   end
   return reshape(cat(imgsA...,dims=4),imsize,imsize,3,length(imgsA)),reshape(cat(imgsB...,dims=4),imsize,imsize,3,length(imgsB))
end

function make_minibatch(X, idxs,size=256)
    """
    size : Image dimension
    """
    X_batch = Array{Float32}(undef, size, size, 3, length(idxs))
    for i in 1:length(idxs)
        X_batch[:,:,:, i] = Float32.(X[:,:,:,idxs[i]])
    end
    return X_batch
end

function nullify_grad!(p)
  if typeof(p) <: TrackedArray
    p.grad .= 0.0f0
  end
  return p
end

function zero_grad!(model)
  model = mapleaves(nullify_grad!, model)
end

function norm(x)
    # convert(CuArray{Float32},2.0 .* x .- 1.0)
    (2.0f0 .* x) .- 1.0f0
end

function denorm(x)
   # convert(CuArray{Float32},((x .+ 1.0)./(2.0) ))
   (x .+ 1.0f0) ./ 2.0f0
end

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

drop_first_two(x) = dropdims(x,dims=(1,2))

function save_to_image(var,name)
  """
  Takes in a variable on the gpu and saves it to an image in a directory
  """
  cpu_out = cpu(denorm(var))
  cpu_out = cpu_out .- minimum(cpu_out)
  cpu_out = cpu_out ./ maximum(cpu_out)
  s = size(cpu_out)
  img = colorview(RGB,reshape(cpu_out[:,:,:,1],3,s[1],s[2]))
  save(string("./sample/",name),img)
end

function get_image_array(var)
  """
  Takes in a variable on the gpu and gives the corresponding processed image array
  """
  cpu_out = cpu(denorm(var))
  cpu_out = cpu_out .- minimum(cpu_out)
  cpu_out = cpu_out ./ maximum(cpu_out)
  s = size(cpu_out)
  reshape(cpu_out[:,:,:,1],3,s[1],s[2])
end

# BatchNorm Wrapper
function BatchNormWrap(out_ch)
    Chain(x->expand_dims(x,2),
    BatchNorm(out_ch),
    x->squeeze(x))
end

# Loss function
# The binary cross entropy loss
function bce(ŷ, y)
    -y.*log.(ŷ .+ 1f-10) - (1  .- y).*log.(1 .- ŷ .+ 1f-10)
end

function logitbinarycrossentropy(logŷ, y)
  mean((1 .- y).*logŷ .- logσ.(logŷ))
end
