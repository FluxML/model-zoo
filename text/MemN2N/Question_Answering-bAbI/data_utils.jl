using Base.Iterators: flatten
using Random

function parse_stories(lines)
      data =[]
      story=[]
      for line in lines
            line = lowercase(line)
            (nid, line) = split(line,' ',limit=2)
            nid = parse(Int,nid)
            line = strip(line, ['.',' '])
            if nid==1
                  story = []
            end
            if occursin("\t", line)
                  (q, a, supporting) = split(line,'\t')
                  q = strip(q, [' ','?'])
                  q = map(x->string(x), split(q))
                  a=[string(a)]
                  push!(data, (story[1:end],q,a))
            else
                  line = map(x->string(x), split(line))
                  push!(story, line)
            end
      end
      return data
end

function createVocab(data)
      words = Set(collect(flatten(flatten(flatten([[a,[b], [c]] for (a,b,c) in data])))))
      words2idx = Dict(j=>i for (i,j) in enumerate(words))
      return words2idx
end

function vectorize_data(data, word2idx, sentence_size, memory_size)
      vectorized_data = []
      for (story, query, answer) in data
            story_vec = map(x->map(y->word2idx[y], x),story)
            if length(story_vec)>memory_size
                  story_vec = story_vec[end-memory_size+1:end]
            end
            time_vec = collect(1:length(story_vec))
            query_vec = map(y->word2idx[y], query)
            answer_vec = zeros(Float64, length(word2idx))
            answer_vec[word2idx[answer[1]]] = 1
            answer_vec = reshape(answer_vec, (1,length(word2idx)))
            push!(vectorized_data, [story_vec, query_vec, answer_vec, time_vec])
      end

      for i=1:length(vectorized_data)
            for j=1:length(vectorized_data[i][1])
                  if length(vectorized_data[i][1][j])<sentence_size
                        append!(vectorized_data[i][1][j], ones(Int64, sentence_size-length(vectorized_data[i][1][j])))
                  end
            end

            if length(vectorized_data[i][2])<sentence_size
                  append!(vectorized_data[i][2], ones(Int64, sentence_size-length(vectorized_data[i][2])))
            end
      end
      return vectorized_data
end

function create_vocab(data)
      vocab = Set(collect(flatten(flatten(flatten([[a,[b], [c]] for (a,b,c) in data])))))
      word2idx = Dict(word=>i+1 for (i,word) in enumerate(vocab))
      word2idx["Nil"] = 1
      return (vocab, word2idx)
end

function create_dataset(data_dir)
      train_data = []
      test_data = []

      for i=1:20
          stub = "qa"*string(i)
          train_fName = filter(x->occursin("train",x)&&occursin(stub, x), readdir(data_dir))[1]
          test_fName = filter(x->occursin("test",x)&&occursin(stub, x), readdir(data_dir))[1]
          push!(train_data, parse_stories(readlines(open(joinpath(data_dir,train_fName)))))
          push!(test_data, parse_stories(readlines(open(joinpath(data_dir,test_fName)))))
      end

      data = collect(flatten(cat(train_data, train_data, dims=1)))
      (vocab, word2idx) = create_vocab(data)

      return (data, vocab, word2idx)
end
