using Pkg, Pkg.TOML

root = joinpath(@__DIR__, "..")

meta = TOML.parsefile(joinpath(@__DIR__, "Notebooks.toml"))
meta = meta[ARGS[1]]

path = meta["path"]
deps = get(meta, "deps", [])

for d in ["Project.toml", "Manifest.toml", ".gitignore"]
  isfile(joinpath(root, path, d)) && push!(deps, d)
end

for dep in deps
  cp(joinpath(root, path, dep), joinpath(root, "notebooks", path, dep), force = true)
end

pushfirst!(LOAD_PATH, @__DIR__)
Pkg.activate(joinpath(root, "notebooks", path))

using Literate

scripts = meta["notebook"]
scripts isa String && (scripts = [scripts])

for script in scripts
  Literate.notebook(joinpath(root, path, script),
                    joinpath(root, "notebooks", path),
                    credit = false)
end
