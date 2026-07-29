using CUDAnative
device!(2)

using Images,CuArrays,Flux
using Flux:@treelike, Tracker
using Base.Iterators: partition
using Random
using Statistics
using Flux.Tracker:update!
using BSON: @save,@load
using Flux:testmode!
using Distributions:Normal

include("utils.jl")
include("generator.jl")
include("discriminator.jl")

BATCH_SIZE = 4
NUM_EXAMPLES = 10
# Define Paths
DATA_PATH = "../data/facades/train"
LOAD_PATH = "../weights/"

function sampleA2B(X_A_test,gen;base_id="1")
    """
    Samples new images in domain B
    X_A_test : N x C x H x W array - Test images in domain A
    """
    X_A_test = norm(X_A_test)
    X_B_generated = denorm(cpu(gen(X_A_test |> gpu)).data)
    
    imgs = []
    s = size(X_B_generated)
    for i in 1:size(X_B_generated)[end]
       xt = reshape(X_A_test[:,:,:,i],256,256,3,1)
       xb = reshape(X_B_generated[:,:,:,i],256,256,3,1)
       out_array = cat(get_image_array(xt),get_image_array(xb),dims=3)
       save(string("../sample/",base_id,"_$i.png"),colorview(RGB,out_array))
    end
    imgs
end

function test()
   # load test data
   data = load_dataset(DATA_PATH,256)[1:NUM_EXAMPLES]
   println(data[1])
   
   # Split into batches
   mb_idxs = partition(1:length(data), BATCH_SIZE)
   train_batches = [data[i] for i in mb_idxs]
    
   @load string(LOAD_PATH,"gen.bson") gen

   gen = gen |> gpu
    
   println("Loaded Generator")

   for i in 1:length(train_batches)
        data_mb,_ = get_batch(train_batches[i],256) |> gpu
        data_mb = reshape(data_mb,256,256,3,BATCH_SIZE) 
        out = sampleA2B(data_mb,gen;base_id=string(i))
   end
end
