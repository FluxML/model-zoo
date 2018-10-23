using Pkg, Pkg.TOML

root = joinpath(@__DIR__, "..")

meta = TOML.parsefile(joinpath(@__DIR__, "Notebooks.toml"))
meta = meta[ARGS[1]]

path = meta["path"]
deps = get(meta, "deps", [])

for d in ["Project.toml", "Manifest.toml", ".gitignore"]
  isfile(joinpath(root, path, d)) && push!(deps, d)
end

mkpath(joinpath(root, "notebooks", path))
for dep in deps
  cp(joinpath(root, path, dep), joinpath(root, "notebooks", path, dep), force = true)
end

pushfirst!(LOAD_PATH, @__DIR__)
Pkg.activate(joinpath(root, "notebooks", path))

using Literate

function init_nb(content)
	content = replace(content, r"#\s*using CuArrays" => "## using CuArrays")
	content = "using Pkg; Pkg.activate(\".\"); Pkg.instantiate();\n\n" * content
	return content
end

scripts = meta["notebook"]
scripts isa String && (scripts = [scripts])

for script in scripts
  Literate.notebook(joinpath(root, path, script),
                    joinpath(root, "notebooks", path),
                    credit = false, preprocess = init_nb)
end

scripts = map(x -> x[1:end - 3] * ".ipynb", scripts)
keep = union(deps, scripts)
files = readdir(joinpath(root, "notebooks", path))

for r in files
  r in keep || rm(joinpath(root, "notebooks", path, r, force = true))
end
