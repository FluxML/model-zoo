using Pkg
Pkg.activate("..")
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
using Distributions:Normal,Uniform

include("utils.jl")
include("generator.jl")
include("discriminator.jl")

gen_A = UNet()
TEST_A_PATH = "../../../../../../cycleGAN/data/trainA/"
TEST_B_PATH = "../../../../../../cycleGAN/data/trainB/"
LOAD_PATH = "../weights/" # For loading the weights
SAVE_PATH = "../sample/" # For saving the generated samples

### SAMPLING ###
function sampleA2B(X_A_test,gen_A)
    """
    Samples new images in domain B
    X_A_test : N x C x H x W array - Test images in domain A
    """
    X_A_test = norm(X_A_test)
    X_B_generated = cpu(denorm(gen_A(X_A_test |> gpu)))
    imgs = []
    s = size(X_B_generated)
    for i in size(X_B_generated)[end]
       push!(imgs,colorview(RGB,reshape(X_B_generated[:,:,:,i],3,s[1],s[2])))
    end
    imgs
end

function test()
   # load test data
   dataA = load_dataset(TEST_A_PATH,256)[:,:,:,2]
   dataA = reshape(dataA,256,256,3,1)
   @load string(LOAD_PATH,"gen_A.bson") gen_A
   gen_A = gen_A |> gpu
   out = sampleA2B(dataA,gen_A)
   for (i,img) in enumerate(out)
        save(string(SAVE_PATH,"A_$i.png"),img)
   end
end

test()

