using Base.Iterators: flatten
using Random

include("data_utils.jl")
include("MemN2N_model_babi.jl")

data_dir = "/Users/dpk/Downloads/tasks_1-20_v1-2/en-10k/"
ids = collect(1:20)

train_data = []; test_data = [];
for i in ids
    global test_data, train_data
    stub = "qa"*string(i)
    train_fName = filter(x->occursin("train",x)&&occursin(stub, x), readdir(data_dir))[1]
    test_fName = filter(x->occursin("test",x)&&occursin(stub, x), readdir(data_dir))[1]
    push!(train_data, parse_stories(readlines(open(joinpath(data_dir,train_fName)))))
    push!(test_data, parse_stories(readlines(open(joinpath(data_dir,test_fName)))))
end

data = collect(flatten(cat(train_data, train_data, dims=1)))

vocab = Set(collect(flatten(flatten(flatten([[a,[b], [c]] for (a,b,c) in data])))))

word2idx = Dict(word=>i+1 for (i,word) in enumerate(vocab))
word2idx["Nil"] = 1

max_story_size = maximum(map(x->length(x), [story for (story,q,a) in data]))
sentence_size_story = maximum(map(x->length(x), flatten([story for (story,q,a) in data])))
query_size =maximum(map(x->length(x), [q for (story,q,a) in data]))
memory_size = min(50, max_story_size)

sentence_size = max(query_size, sentence_size_story)
vocab_size = length(word2idx)

data = shuffle!(vectorize_data(data, word2idx, sentence_size, memory_size))
train_dataset = data[1:18000]
test_dataset = data[18001:end]

mem = create_memory(vocab_size, 40,0,memory_size , 0.01, 0.01, 3, sentence_size)

train(train_dataset, mem, 40, 1024, 10, 0.1)
