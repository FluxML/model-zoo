"""
Takes in t, which is an array of tuples of (`sequence`, `target`) where `sequence` is an array of timestamp x features, 
and target is either an array or a single value. Outputs a tuple of batched 

julia> z = [([1 2 3; 2 3 4], 5), ([6 7 8; 7 8 9], 0)]
2-element Array{Tuple{Array{Int64,2},Int64},1}:
 ([1 2 3; 2 3 4], 5)
 ([6 7 8; 7 8 9], 0)

julia> batch_ts(z)
([1 2 3; 2 3 4]

[6 7 8; 7 8 9], [5]

[0])

julia> size(batch_ts(z)[1])
(2, 3, 2)

julia> size(batch_ts(z)[2])
(1, 1, 2)
"""
batch_ts(t) = reduce((x, y) -> (cat(x[1], y[1], dims=3), cat(x[2], y[2], dims=3)), t)
batch_ts(t::Tuple) = (unsqueeze(t[1],3), unsqueeze(t[2],3)) # handle batch size of 1