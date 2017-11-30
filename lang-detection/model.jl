using Flux
using Flux: onehot, onehotbatch, crossentropy, reset!, throttle

corpora = Dict()

cd(@__DIR__)
for file in readdir("corpus")
  lang = Symbol(match(r"(.*)\.txt", file).captures[1])
  corpus = filter(!isempty, split(String(read("corpus/$file")), "."))
  corpus = strip.(normalize_string.(corpus, casefold=true, stripmark=true))
  corpora[lang] = corpus
end

langs = collect(keys(corpora))
alphabet = ['a':'z'; '0':'9'; ' '; '\n'; '_']

# See which chars will be represented as "unknown"
unique(filter(x -> x ∉ alphabet, join(vcat(values(corpora)...))))

dataset = [(onehotbatch(s, alphabet, '_'), onehot(l, langs))
           for l in langs for s in corpora[l]] |> shuffle

train, test = dataset[1:end-100], dataset[end-99:end]

N = 15

scanner = Chain(Dense(length(alphabet), N), LSTM(N, N))
encoder = Dense(N, length(langs))

function model(x)
  state = scanner.(x.data)[end]
  reset!(scanner)
  softmax(encoder(state))
end

loss(x, y) = crossentropy(model(x), y)

testloss() = mean(loss(t...) for t in test)

opt = ADAM(params(scanner, encoder))
evalcb = () -> @show testloss()

Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))

open(io -> serialize(io, (langs, alphabet, scanner, encoder)), "model.jls", "w")
