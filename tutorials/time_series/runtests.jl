using Test
using DataFrames
using Random
using Dates

rng = MersenneTwister(123)

times = 150
df = DataFrame("timestamp"=>1:times, 
               "A"=>randn(rng, Float16, times), 
               "B"=>sin.(rand(rng, times)),
               "C"=>1:times)
train_prop = 0.8
train_end = Int(times*train_prop)
train_df = df[1:train_end,:]
valid_df = df[train_end+1:end,:]


# WindowGenerator
include("window_generator.jl")
@testset "historical window of length 1" begin
    h = 1
    f = 1
    wg = WindowGenerator(h, f, train_df, valid_df, "C")
    @test wg.target_idx == [4]
    @test size(wg.train[1][1]) == (4,1)
    @test size(wg.valid[1][1]) == (4,1)
    @test length(wg.train) == 119
    @test length(wg.valid) == 29
    @test (wg.train[1][1][1]) == (wg.train[1][2][1] - 1)
    @test (wg.train[end][1][1]) == (wg.train[end][2][1] - 1)
end


@testset "historical window of length 10" begin
    
end


@testset "multiple label columns" begin
    
end


# batch_ts
include("batch_ts.jl")
@testset "" begin
    
end