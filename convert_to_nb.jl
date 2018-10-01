using Literate

function find_notebooks()
  for (root, dirs, files) in walkdir(".")
    for file in files
      if endswith(file, ".jl") || endswith(file, ".ipynb")
        println(joinpath(root, file))
      end
    end
  end
end

function convert_to_nb(input_script = "", output_dest = "output")
  if input_script == ""
    return
  end
  if !isdir(output_dest)
    mkdir(output_dest)
  end
  Literate.notebook(input_script, output_dest, execute = true)
end

function main(output_dest = "output")
  input_script = ARGS[1]
  if length(ARGS) > 1
    output_dest = ARGS[2]
  end
  convert_to_nb(input_script, output_dest)
end

main()