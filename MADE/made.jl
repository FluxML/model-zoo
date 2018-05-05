#=
Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
Re-implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509
Re-Re-implementation by Tejan Karmali using Flux.jl
=#

using Flux
using Flux: glorot_uniform, back!

# ------------------------------------------------------------------------------

add_dims_r(a) = reshape(a, size(a)..., 1)
add_dims_l(a) = reshape(a, 1, size(a)...)

struct MaskedDense{F,S,T,M}
  # same as Linear except has a configurable mask on the weights

  W::S
  b::T
  mask::M
  σ::F
end

function MaskedDense(in::Integer, out::Integer; σ = identity)
  return MaskedDense(param(glorot_uniform(out, in)), param(zeros(out)), ones(out, in), σ)
end

function (a::MaskedDense)(x)
  W, b, mask, σ = a.W, a.b, a.mask, a.σ
  σ.(mask .* W * x .+ b)
end

function set_mask(a::MaskedDense, mask)
  a.mask = mask
end

mutable struct MADE
  #=
    nin: integer; number of inputs
    hidden sizes: a list of integers; number of units in hidden layers
    nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
          note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
          will be all the means and the second nin will be stds. i.e. output dimensions depend on the
          same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
          the output of running the tests for this file makes this a bit more clear with examples.
    num_masks: can be used to train ensemble over orderings/connections
    natural_ordering: force natural ordering of dimensions, don't use random permutations
  =#

  nin::Integer
  nout::Integer
  hidden_sizes::Array{Integer, 1}
  net::Chain

  # seeds for orders/connectivities of the model ensemble
  natural_ordering::Bool
  num_masks
  seed::UInt  # for cycling through num_masks orderings

  m::Dict

  function MADE(in::Integer, hs, out::Integer, nat_ord::Bool, num_masks = 1)
    # define a simple MLP neural net
    hs = push!([in], hs..., out)
    layers = [MaskedDense(hs[i], hs[i + 1]; σ = relu) for i = 1:length(hs) - 2]

    net = Chain(layers..., MaskedDense(hs[end-1], hs[end]))

    new(in, out, hs[2:end-1], net, nat_ord, 1, 0, Dict())
  end
end

function update_masks(made::MADE)
  if made.m != Dict() && made.num_masks == 1
    return # only a single seed, skip for efficiency
  end

  L = length(made.hidden_sizes)

  # fetch the next seed and construct a random stream
  rng = MersenneTwister(made.seed)
  made.seed = (made.seed + 1) % made.num_masks

  # sample the order of the inpdimensionsuts and the connectivity of all neurons
  made.m[0] = made.natural_ordering ? collect(1:made.nin) : randperm(rng, made.nin)
  for l = 1:L
    made.m[l] = rand(rng, minimum(made.m[l - 1]):made.nin - 2, made.hidden_sizes[l])
  end

  # construct the mask matrices
  masks = [add_dims_r(made.m[l - 1]) .<= add_dims_l(made.m[l]) for l = 1:L]
  push!(masks, add_dims_r(made.m[L]) .< add_dims_l(made.m[0]))

  # handle the case where nout = nin * k, for integer k > 1
  if made.nout > made.nin
    k = div(made.nout, made.nin)
    # replicate the mask across the other outputs
    masks[end] = hcat(tuple((masks[end] for i=1:k)...))
  end

  # set the masks in all MaskedLinear layers
  for (l, m) in zip(made.net, masks)
    typeof(m) == MaskedDense ? set_mask(l, m) : continue
  end
end

function (made::MADE)(x)
  made.net(x)
end
# ------------------------------------------------------------------------------

#Main

# run a quick and dirty test for the autoregressive property
D = 10
rng = MersenneTwister(14)
x = convert.(Float32, rand(rng, D, 1) .> 0.5)
configs = [
  (D, [], D, false),                 # test various hidden sizes
  (D, [200], D, false),
  (D, [200, 220], D, false),
  (D, [200, 220, 230], D, false),
  (D, [200, 220], D, true),          # natural ordering test
  (D, [200, 220], 2 * D, true),      # test nout > nin
  (D, [200, 220], 3 * D, false)      # test nout > nin
  ]

for (nin, hiddens, nout, natural_ordering) in configs
  println("checking nin $nin, hiddens $hiddens, nout $nout, natural $natural_ordering")
  model = MADE(nin, hiddens, nout, natural_ordering)
  update_masks(model)
  # run backpropagation for each dimension to compute what other
  # dimensions it depends on.
  res = []
  for k = 1:nout
    xtr = param(x)
    xtrhat = model(xtr)
    loss = xtrhat[k, 1]
    back!(loss)

    depends = xtr.grad[1] .!= 0
    depends_ix = find(depends)
    isok = k % nin + 1 ∉ depends_ix
    push!(res, (length(depends_ix), k, depends_ix, isok))
  end

  # pretty print the dependencies
  sort!(res, by = x -> x[2])
  for (nl, k, ix, isok) in res
    print("output $k depends on inputs: $ix : ")
    println(isok ? "OK" : "NOTOK")
  end
end
