using Pkg.TOML
meta = TOML.parsefile(joinpath(@__DIR__, "Notebooks.toml"))

for proj in keys(meta)
  run(`julia1 convert.jl $proj`)
end
