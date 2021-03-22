using Test
using DataFrames
using Random
using Dates

include("window_generator.jl")
include("batch_ts.jl")

rng = MersenneTwister(123)

df = DataFrame("timestamp"=>Date(2014,11,15):Day(1):Date(2014, 12, 31), 
               "A"=>randn(rng, Float16, 47), 
               "B"=>sin.(rand(rng, 47)))
num_train = 35
train_df = df[1:num_train,:]
test_df = df[num_train+1:end,:]

# WindowGenerator
@testset "historical window of length 1" begin
    
end


@testset "multiple label columns" begin
    
end


@testset "" begin
    
end


# batch_ts
@testset "" begin
    
end


WindowGenerator(6, 1, train_df, valid_df, label_columns=["T (degC)"])
WindowGenerator(6, 1, train_df, valid_df, label_columns=["T (degC)", "Wx"])