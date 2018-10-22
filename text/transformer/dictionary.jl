# This script extracts a dictionary from the europarl data.
# It need only be run once, as the dictionary is saved in the `data` folder.

using WordTokenizers, DelimitedFiles

corpus(name) = split(String(read(name)), "\n")

english = corpus(joinpath(@__DIR__, "data", "europarl-v7.fr-en.en"))
french  = corpus(joinpath(@__DIR__, "data", "europarl-v7.fr-en.fr"))

function frequencies(corpus)
  fs = Dict{String,Int}()
  for s in corpus, t in tokenize(s)
    t = lowercase(t)
    fs[t] = get(fs, t, 0) + 1
  end
  return fs
end

en_fs = frequencies(english)
fr_fs  = frequencies(french)

dict(fs) = sort(collect(keys(fs)), by = k -> -fs[k])[1:10_000]

dict(en_fs)
dict(fr_fs)

writedlm(joinpath(@__DIR__, "data", "dict.en"), dict(en_fs))
writedlm(joinpath(@__DIR__, "data", "dict.fr"), dict(fr_fs))
