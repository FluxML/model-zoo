using Flux: onecold
using StatsBase: wsample

include("1-model.jl")

function detokenise(seq)
  s = Symbol[]
  for c in map(t -> wsample(phones, t), seq)
    c == :end && break
    push!(s, c)
  end
  return join(s, " ")
end

function predict(m, s)
  ŷ = m(Batch([tokenise(s, alphabet) for i = 1:50]))[1]
  detokenise(ŷ)
end

# predict(mxmodel, "SWIPES")
# predict(mxmodel, "ARM")
