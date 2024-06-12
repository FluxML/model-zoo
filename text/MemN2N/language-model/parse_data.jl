import Base.Iterators: flatten

function createVocab(fileName::String, idx2word, word2idx)
    """
    Takes a txt file and returns data as word indices
    Arguments:
        fileName::String: path to txt file
        idx2word::Array: Empty Dict that will have mappings from indices to words
        word2idx::Dict: Empty dict that will have mappings from words to indices
    """
    data = []
    lines=[]
    if isfile(fileName)
        open(fileName, "r") do fstream
            lines = readlines(fstream)
        end
    else
        error("File not found!!")
    end

    words = collect(Set(collect(flatten([split(line) for line in lines]))))
    if length(idx2word)==0
        idx2word[1]="<eos>"
        word2idx["<eos>"]=1
    end

    for word in words
        if !haskey(word2idx, word)
            word2idx[word]=length(word2idx)+1
            idx2word[length(word2idx)] = word
        end
    end

    for line in lines
        for word in split(line)
            push!(data, word2idx[word])
        end
        push!(data, word2idx["<eos>"])
    end
    println("Read $(length(data)) words from $(basename(fileName))")
    return data
end
