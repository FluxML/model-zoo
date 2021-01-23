using Pkg, Pkg.TOML

root = joinpath(@__DIR__, "..")

meta = TOML.parsefile(joinpath(@__DIR__, "Notebooks.toml"))
meta = meta[ARGS[1]]

path = meta["path"]
deps = get(meta, "deps", [])
deps = deps isa String ? [deps] : deps

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

function postprocess_nb(content)
  content = replace(content, r"\s*using CUDA" => "## using CUDA")
  return content
end

function preprocess_nb(content)
  content = replace(content, r"#\s*using CUDA" => "using CUDA")
  content = "using Pkg; Pkg.activate(\".\"); Pkg.instantiate();\n\n" * content
  return content
end

function init_nb(content)
  content = "using Pkg; Pkg.activate(\"$root\"); Pkg.status();\n\n" * content
  return content
end

scripts = meta["notebook"]
scripts isa String && (scripts = [scripts])

for script in scripts
  Literate.notebook(joinpath(root, path, script),
                    joinpath(root, "notebooks", path),
                    credit = false, preprocess = preprocess_nb,
                    postprocess = postprocess_nb)
end

scripts = map(x -> x[1:end - 3] * ".ipynb", scripts)
nbs = filter(x -> endswith(x, ".ipynb"), readdir(joinpath(root, path)))
keep = union(deps, scripts, nbs)
files = readdir(joinpath(root, "notebooks", path))

for r in files
  r in keep || rm(joinpath(root, "notebooks", path, r), force = true)
end
