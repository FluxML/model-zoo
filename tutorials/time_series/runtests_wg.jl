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


include("window_generator.jl")
@testset "historical window of length 1, future window 1" begin
    h = 1
    f = 1
    wg = WindowGenerator(h, f, train_df, valid_df, "C")
    @test wg.target_idx == [4]
    @test size(wg.train[1][1]) == (length(names(df)),h)
    @test size(wg.valid[1][1]) == (length(names(df)),h)
    @test length(wg.train) == train_end - (h+f-1) # truncates necessary first and last sequence points
    @test length(wg.valid) == times - train_end - (h+f-1)
    @test (wg.train[1][1][1]) == (wg.train[1][2][1] - h)
    @test (wg.train[end][1][1]) == (wg.train[end][2][1] - h)
    @test wg.train[end][1][1,end] == (wg.train[end][2][1,begin] - 1) # 1st pred is immediately after last hist

end

@testset "historical window of length 7, future window 5" begin
    h = 7
    f = 5
    wg = WindowGenerator(h, f, train_df, valid_df, "C")
    @test wg.target_idx == [4]
    @test size(wg.train[1][1]) == (length(names(df)),h)
    @test size(wg.valid[1][1]) == (length(names(df)),h)
    @test length(wg.train) == train_end - (h+f-1)
    @test length(wg.valid) == times - train_end - (h+f-1)
    @test (wg.train[1][1][1]) == (wg.train[1][2][1] - h)
    @test (wg.train[end][1][1]) == (wg.train[end][2][1] - h)
    @test wg.train[end][1][1,end] == (wg.train[end][2][1,begin] - 1) # 1st pred is immediately after last hist
end

@testset "multiple label columns" begin
    h = 16
    f = 3
    wg = WindowGenerator(h, f, train_df, valid_df, "C")
    @test length(wg.target_idx) == 1
    wg = WindowGenerator(h, f, train_df, valid_df, ["A","C"])
    @test length(wg.target_idx) == 2
    wg = WindowGenerator(h, f, train_df, valid_df, ["A","C","B"])
    @test length(wg.target_idx) == 3
    wg = WindowGenerator(h, f, train_df, valid_df, ["timestamp","A","C","B"])
    @test length(wg.target_idx) == 4
end

@testset "ignores nonexistent/duplicate columns" begin
    h = 5
    f = 3
    wg = WindowGenerator(h, f, train_df, valid_df, ["timestamp","A","C","B","D"])
    @test length(wg.target_idx) == 4

    wg = WindowGenerator(h, f, train_df, valid_df, ["timestamp","timestamp"])
    @test length(wg.target_idx) == 1
end

@testset "labels_indices are correct" begin
    valid_end = times - train_end # limited by size of validation set
    for h in 1:(valid_end - 1)
        for f in 1:(valid_end - h)
            wg = WindowGenerator(h, f, train_df, valid_df, ["timestamp","A","C","B"])
            @test f == length(wg.label_indices)
            @test first(wg.label_indices) == h + 1
            @test last(wg.label_indices) == h + f
        end
    end
end




# batch_ts
include("batch_ts.jl")
@testset "" begin
    
end