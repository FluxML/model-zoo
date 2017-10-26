using JLD
using Flux
using Images

function savemodel(filename, m)
  p = map(x -> Array(x.data), params(m))
  save(filename, "p", p)
end

function restoremodel!(filename, m)
  p = load(filename, "p")

  for (param, val) ∈ zip(params(m), p)
    param.data[:] = val
  end

  m
end

function outputimg(filename, x)
  out = reshape(m(x).data .* 255, 28, 28)
  save(filename, map(clamp01nan, colorview(Gray, out)))
end

function outputimgs(filename, xs)
  xs′ = mapslices(xs, 1) do x
    out = reshape(m(x).data .* 255, 28, 28)
    [map(clamp01nan, colorview(Gray, out))]
  end

  orig = mapslices(xs, 1) do x
    [colorview(Gray, map(clamp01nan, reshape(x, 28, 28)))]
  end
  orig = reduce(hcat, orig)
  gen = reduce(hcat, xs′)

  save(filename, [orig; gen])
end
