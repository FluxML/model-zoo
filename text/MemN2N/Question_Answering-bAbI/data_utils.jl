using Base.Iterators: flatten

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

function vectorize_data(data, word_idx, sentence_size, memory_size)
      vectorized_data = []
      for (story, query, answer) in data
            story_vec = map(x->map(y->word2idx[y], x),story)
            if length(story_vec)>memory_size
                  story_vec = story_vec[end-memory_size+1:end]
            end
            time_vec = collect(1:length(story_vec))
            query_vec = map(y->word2idx[y], query)
            answer_vec = zeros(Float64, length(word_idx))
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
