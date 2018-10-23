using Pkg
Pkg.activate(@__DIR__)

using Pkg.TOML
meta = length(ARGS) > 0 ? ARGS :
  keys(TOML.parsefile(joinpath(@__DIR__, "Notebooks.toml")))

convertjl = joinpath(@__DIR__, "convert.jl")

for proj in meta
  run(`$(Base.julia_cmd()) $convertjl $proj`)
end
