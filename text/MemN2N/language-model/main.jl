using ArgParse
include("parse_data.jl")
include("MemN2N_model.jl")


function parse_command_line()
	s = ArgParseSettings()

	@add_arg_table s begin
        "path"
            help = "path to training dataset"
            required = true
        "--edim"
            help = "internal state dimension"
            arg_type = Int
            default = 150
        "--lindim"
            help = "linear part of the state"
            arg_type = Int
            default = 75
        "--nhops"
            help = "Number of hops"
            arg_type = Int
            default = 6
        "--init_hid"
            help = "initial internal state value"
            arg_type = Float64
            default = 0.01
        "--init_std"
            help = "weight initialization std"
            arg_type = Float64
            default = 0.01
        "--batch_size"
            help = "batch size to use during training"
            arg_type = Int
            default = 128
        "--mem_size"
            help = "memory size"
            arg_type = Int
            default = 128
        "--epochs"
            help = "number of epochs"
            arg_type = Int
            default = 50
        "--init_lr"
        	help = "initial learning rate"
        	arg_type = Float64
        	default = 0.01
        "--max_grad_norm"
        	help = "Maximum value to which gradient norm should be clipped"
        	arg_type = Int
        	default = 50
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_command_line()
    idx2word = Dict()
	word2idx = Dict()
	train_data = createVocab(parsed_args["path"], idx2word, word2idx)
	num_words = length(word2idx)
	data = Data(train_data, num_words)
	memory = create_memory(num_words, parsed_args["edim"], parsed_args["lindim"],
							parsed_args["mem_size"], parsed_args["init_hid"],
							parsed_args["init_std"], parsed_args["nhops"])
	train(data, memory, parsed_args["epochs"], parsed_args["batch_size"], parsed_args["max_grad_norm"], parsed_args["init_lr"])

end

main()
