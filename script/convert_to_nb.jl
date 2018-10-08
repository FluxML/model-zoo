using Pkg, Literate
import TOML

function activate_env(path)
  files = readdir(path)
  "Project.toml" in files && Pkg.activate(path)
end

function convert_to_nb(input_script = "", 
                      output_dest = "output"; 
                      name = input_script[1:end-3])
  if input_script == ""
    return
  end
  if !isdir(output_dest)
    mkpath(joinpath(output_dest, "tmp")) # force mkpath to create a dir
    rm(joinpath(output_dest, "tmp"))
  end
  Literate.notebook(input_script, output_dest, execute = true, name = name)
end

# TODO:

# 1.* Get a root directory (relative path, starting from top directory in repo)
# 2.* Look for a Conf.toml in the root
# 3.* Activate the environment in the root
# 4.* Take the file to be made into the notebook, and all deps that need to be copied for the same -> comes from Conf.toml
# 5.* Convert to notebook and output
# 6.* Remove the unnecessary files
# 7.* Ensure correct environment activates for the notebook while executing

# Target File Structure: 

# model-zoo:
#           other
#                 Project.toml
#                 Manifest.toml
#                 xor.jl
          
#           vision
#                 cifar
#                       Project.toml
#                       Manifest.toml
#                       utils.jl
#                       cifar.jl

#                 mnist
#                       Project.toml
#                       Manifest.toml
#                       mlp.jl
#                       vae.jl
#           ...

#           scripts
#                   Manifest.toml
#                   Project.toml
#                   convert_to_nb.jl

#           output
#                 other
#                       Project.toml
#                       Manifest.toml
#                       output_nb.ipynb
#                       ...

#                 vision
#                       cifar
#                             Manifest.toml
#                             Project.toml
#                             utils.jl
#                             output_nb.ipynb
#                             ...

#                       mnist
#                             Manifest.toml
#                             Project.toml
#                             output_nb.ipynb
#                             ...

function get_config(path_to_conf)
  confFile = TOML.parsefile(joinpath(path_to_conf, "Conf.toml"))
  projects = confFile["Project"]
  for project in projects
    root = project["root"]
    confs = project["conf"]
    for conf in confs

      # default behaviour is to take all files in dir
      # as deps
      deps = ["Project.toml", "Manifest.toml"]
      if haskey(conf, "deps")
        conf["deps"] = push!(deps, conf["deps"]...)
      else
        conf["deps"] = readdir(joinpath("..", root))
      end

      # Literate.jl needs name of output file w/o extensions
      if !haskey(conf, "name")
        conf["name"] = conf["script"][1:end-3] # remove ".jl"
      end
    end
  end
  projects
end

# everytihng in output dir = nb + deps + extras
# root = "vision/mnist"
# script = "mlp.jl"
# deps = ["vae.jl"]

function main(output = "output")
  project_dir = ARGS[1]
  if length(ARGS) > 1
    output = ARGS[2]
  end

  @assert "Project.toml" in readdir(project_dir) && "Manifest.toml" in readdir(project_dir)
  projects = get_config(project_dir)
  cd("..")

  for project in projects
    root, confs = project["root"], project["conf"]
    
    keep = [] # keep track of files that need to be output
    for conf in confs
      
      deps, script = conf["deps"], conf["script"]
      push!(keep, deps...)
      push!(keep, conf["name"] * ".ipynb")
      activate_env(root)

      convert_to_nb(joinpath(root, script),
                    joinpath(output, root);
                    name = conf["name"])

      # copy deps
      for dep in deps
        src = joinpath(root, dep)
        dest = joinpath(output, root * "/" * dep)
        cp(src, dest, force = true)
      end
    end

    # remove unnecessary files
    output_files = readdir(joinpath(output, root))
    foreach(x -> rm(joinpath(output, root, x)), filter(x -> !in(x, keep), output_files))
  end
end

main()