# interface stuffs

using Flux: back!
using Flux.Optimise: runall

using LearningStrategies: LearningStrategy
import LearningStrategies: setup!, hook, update!, finished

struct FluxModel <: LearningStrategy
  opt
  cb
  FluxModel(m, opt, cb = () -> ()) = new(runall(opt(params(m))), runall(cb))
end

function update!(loss, a::FluxModel, data)
  l = loss(data...)
  isinf(l) && error("Loss is Inf")
  isnan(l) && error("Loss is NaN")
  back!(l)
  a.opt()
end

finished(a::FluxModel, loss, i) = (a.cb() == :stop)
