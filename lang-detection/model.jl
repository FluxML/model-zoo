using Flux
using Flux: onehot, onehotbatch, crossentropy, reset!, throttle

corpora = Dict()

cd(@__DIR__)
for file in readdir("corpus")
  lang = Symbol(match(r"(.*)\.txt", file).captures[1])
  corpora[lang] = filter(!isempty, split(String(read("corpus/$file")), "."))
end

langs = collect(keys(corpora))
alphabet = ['A':'Z'; 'a':'z'; '0':'9'; ' '; '\n'; '_']

# See which chars will be represented as "unknown"
unique(filter(x -> x ∉ alphabet, join(vcat(values(corpora)...))))

dataset = [(onehotbatch(s, alphabet, '_'), onehot(l, langs))
           for l in langs for s in corpora[l]] |> shuffle

scanner = LSTM(length(alphabet), 300)
encoder = Dense(300, length(langs))

function model(x)
  state = scanner.(x.data)[end]
  reset!(scanner)
  softmax(encoder(state))
end

loss(x, y) = crossentropy(model(x), y)

evalcb = () -> @show model(dataset[1][1])

opt = ADAM(params(scanner, encoder))

Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))

open(io -> serialize(io, (alphabet, scanner, encoder)), "model.jls", "w")
