using Flux: onecold

include("1-model.jl")

function detokenise(seq)
  s = Symbol[]
  for c in map(t -> onecold(t, phones), seq)
    c == :end && break
    push!(s, c)
  end
  return join(s, " ")
end

function predict(m, s)
  ŷ = convert(Batch{Seq}, m(Batch([tokenise(uppercase(s), alphabet) for i = 1:50])))[1]
  detokenise(ŷ)
end

predict(mxmodel, "arm")
predict(mxmodel, "hello")
predict(mxmodel, "john")
predict(mxmodel, "viral")

predict(mxmodel, "averse")
predict(mxmodel, "phoneme")
predict(mxmodel, "chammy")

predict(mxmodel, "lice")
predict(mxmodel, "lick")
