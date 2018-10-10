using Pkg.TOML
meta = TOML.parsefile(joinpath(@__DIR__, "Notebooks.toml"))

length(ARGS) > 0 && (meta = ARGS)
meta isa Dict && (meta = keys(meta))

for proj in meta
  run(`julia convert.jl $proj`)
end
