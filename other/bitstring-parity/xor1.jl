include("data.jl")
using Flux, Statistics
using Flux: onehot, onehotbatch, throttle, crossentropy, reset!, onecold

const epochs = 20

train = gendata(100, 2)
val = gendata(10, 2)

scanner = LSTM(length(alphabet), 20)
encoder = Dense(20, length(alphabet))

function model(x)
    state = scanner.(x.data)[end]
    reset!(scanner)
    softmax(encoder(state))
end

loss(x, y) = crossentropy(model(x), y)
batch_loss(data) = mean(loss(d...) for d in data)

opt = ADAM()
ps = params(scanner, encoder)
evalcb = () -> @show batch_loss(val)

for i=1:epochs
    Flux.train!(loss, ps, train, opt, cb=throttle(evalcb, 10))
end

# sanity test
tx = map(c -> onehotbatch(c, alphabet), [
    [false, true], # 01 -> 1
    [true, false], # 10 -> 1
    [false, false], # 00 -> 0
    [true, true]]) # 11 -> 0
[onecold(model(x)) - 1 for x in tx] |> println
