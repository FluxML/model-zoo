module CmdParser
export parse_commandline
using ArgParse

# Argument parsing
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--epochs","-e"
            help = "epoch number, default=100"
            arg_type = Int
            default = 100
        "--batch","-b"
            help = "mini-batch size, default=128"
            arg_type = Int
            default = 50
        "--seq","-s"
            help = "sequence size, default=50"
            arg_type = Int
            default = 50            
        "--gpu","-g"
            help = "gpu index to use, 0,1,2,3,..., default=0"
            arg_type = Int
            default = 1
        "--model","-m"
            help = "use saved model file, default=true"
            arg_type = Bool
            default = true
        "--log","-l"
            help = "create log file, default=true"
            arg_type = Bool
            default = false
    end
    return parse_args(s)
end
end