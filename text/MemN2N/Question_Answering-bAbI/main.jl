using ArgParse

include("data_utils.jl")
include("MemN2N_model_babi.jl")

function parse_command_line()
	s = ArgParseSettings()

	@add_arg_table s begin
        "path"
            help = "path to training dataset"
            required = true
        "--edim"
            help = "internal state dimension"
            arg_type = Int
            default = 40
        "--lindim"
            help = "linear part of the state"
            arg_type = Int
            default = 0
        "--nhops"
            help = "Number of hops"
            arg_type = Int
            default = 3
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
            default = 32
        "--mem_size"
            help = "memory size"
            arg_type = Int
            default = 50
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

	data_dir = parsed_args["path"]
	(data, vocab, word2idx) = create_dataset(data_dir)

	max_story_size = maximum(map(x->length(x), [story for (story,q,a) in data]))
	sentence_size_story = maximum(map(x->length(x), flatten([story for (story,q,a) in data])))
	query_size =maximum(map(x->length(x), [q for (story,q,a) in data]))
	memory_size = min(parsed_args["mem_size"], max_story_size)

	sentence_size = max(query_size, sentence_size_story)
	vocab_size = length(word2idx)

	data = shuffle!(vectorize_data(data, word2idx, sentence_size, memory_size))
	train_dataset = data[1:18000]
	test_dataset = data[18001:end]

	edim = parsed_args["edim"]
	ldim = parsed_args["lindim"]
	init_hid = parsed_args["init_hid"]
	init_std = parsed_args["init_std"]
	nhops = parsed_args["nhops"]

	mem = create_memory(vocab_size, edim, ldim, memory_size , init_hid, init_std, nhops, sentence_size)

	epochs = parsed_args["epochs"]
	batch_size = parsed_args["batch_size"]
	max_norm = parsed_args["max_grad_norm"]
	η = parsed_args["init_lr"]

	train(train_dataset, mem, epochs, batch_size, max_norm, η)
end

main()
