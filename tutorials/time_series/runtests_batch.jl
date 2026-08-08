using Test

include("batch_ts.jl")

@testset "multi batch - single element y" begin
    for _ in 1:100
        dim1 = rand(1:20)
        dim2 = rand(1:20)
        num = rand(2:100) # single item batch not used with single element

        tups = [(rand(dim1, dim2), rand()) for i in 1:num]
        b_tups = batch_ts(tups)
        @test b_tups |> length == 2
        @test size(b_tups[1]) == (dim1, dim2, num)
        @test size(b_tups[2]) == (1, 1, num)
    end
end

@testset "multi batch - multi element y" begin
    for _ in 1:100
        dim1 = rand(1:20)
        dim2 = rand(1:20)
        dim3 = rand(1:20)
        dim4 = rand(1:20)
        num = rand(2:100) # single item batch

        tups = [(rand(dim1, dim2), rand(dim3, dim4)) for i in 1:num]
        b_tups = batch_ts(tups)
        @test b_tups |> length == 2
        @test size(b_tups[1]) == (dim1, dim2, num)
        @test size(b_tups[2]) == (dim3, dim4, num)
    end
end

@testset "single item batch" begin
    for _ in 1:100
        dim1 = rand(1:20)
        dim2 = rand(1:20)
        
        tups = (rand(dim1, dim2), rand(dim1, 1))
        b_tups = batch_ts(tups)
        @test ndims(b_tups[1]) == ndims(b_tups[2]) == 3
    end
end

