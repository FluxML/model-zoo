using Flux: onehot, onehotbatch
using Random

const alphabet = [false, true]  # 0, 1

parity(x) = reduce(xor, x)

gendata(n::Int, k::Int) = gendata(n, k:k)

function gendata(n::Int, k::UnitRange{Int})
    X = bitrand.(rand(k, n))
    return [(onehotbatch(x, alphabet), onehot(y, alphabet)) for (x, y) in zip(X, parity.(X))]
end
