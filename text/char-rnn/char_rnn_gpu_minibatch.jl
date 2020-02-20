using BSON
using BSON: @save,@load
using Flux
using Flux: onehot, chunk, batchseq, throttle, crossentropy
using StatsBase: wsample
using Base.Iterators: partition
using CuArrays
using CUDAnative: device!
using Random
using Dates
using Logging

ϵ = 1.0f-32
working_path = dirname(@__FILE__)
file_path(file_name) = joinpath(working_path,file_name)
include(file_path("cmd_parser.jl"))

model_file = file_path("char_rnn_gpu_minibatch.bson")

# # Get arguments
parsed_args = CmdParser.parse_commandline()
epochs = parsed_args["epochs"]
batch_size = parsed_args["batch"]
use_saved_model = parsed_args["model"]
gpu_device = parsed_args["gpu"]
create_log_file = parsed_args["log"]
sequence = parsed_args["seq"]

if create_log_file
  log_file ="./char_rnn_gpu_minibatch_$(Dates.format(now(),"yyyymmdd-HHMMSS")).log"
  log = open(log_file, "w+")
else
  log = stdout
end
global_logger(ConsoleLogger(log))

start_time = now()
@info "Start - $(start_time)";flush(log)
@info "=============== Arguments ==============="
@info "epochs=$(epochs)"
@info "batch_size=$(batch_size)"
@info "use_saved_model=$(use_saved_model)"
@info "gpu_device=$(gpu_device)"
@info "sequence=$(sequence)"
@info "log_file=$(create_log_file)"
@info "=========================================";flush(log)


device!(gpu_device)
CuArrays.allowscalar(false)

input_file = file_path("input.txt")
isfile(input_file) ||
    download("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
             input_file)

# read(input_file) : 파일에서 텍스트 읽오옴 - 바이너리
# String(read(input_file)) : 바이너리를 스트링으로 변환
# collect(String(read(input_file)) : 스트링을 개별 char array로 변환 - Array{Char,1}
text = collect(String(read(input_file)))

# unique(text) : text에서 unique한 char array를 만든다 - 중복제거 - 하고
# 맨뒤에 '_' 를 추가 한다.
# unique한 char -알파벳 array를 만든다.
alphabet = [unique(text)...,'_']
# ch onehot을 만든다. onhot의 길이는 length(alphabet)이고 onehot에서 1이 있는 위치는
# alphabet에서 ch가 있는 위치와 동일
text = map(ch -> Float32.(onehot(ch,alphabet)),text)
stop = Float32.(onehot('_',alphabet))

N = length(alphabet)
seqlen = sequence
nbatch = batch_size

Xs = collect(partition(batchseq(chunk(text, nbatch), stop), seqlen))
txt = circshift(text,-1)
txt[end] = stop
Ys = collect(partition(batchseq(chunk(txt, nbatch), stop), seqlen))

vloss=Inf; epoch = 0; t_sec = Second(0);
if use_saved_model && isfile(model_file) && filesize(model_file) > 0
  # flush : 버퍼링 없이 즉각 log를 파일 또는 console에 write하도록 함
  @info "Load saved model $(model_file) ...";flush(log)
  # model : @save시 사용한 object명
  @load model_file model vloss epoch sec
  t_sec = sec 
  m = model |> gpu
  run_min = round(Second(t_sec), Minute)
  @info " -> loss : $(vloss), epochs : $(epoch), run time : $(run_min)";flush(log)
else
  @info "Create new model ...";flush(log)  
  model = Chain(
    LSTM(N, 128),
    LSTM(128, 256),
    LSTM(256, 128),
    Dense(128, N),
    softmax)
    m = model |>gpu
end

opt = ADAM(0.01)
tx, ty = (Xs[1]|>gpu, Ys[1]|>gpu)

function loss(xx, yy)
  out = 0.0f0
  for (idx, x) in enumerate(xx)
    out += crossentropy(m(x) .+ ϵ, yy[idx])
  end  
  Flux.reset!(m)
  out
end

@info "Training model...";flush(log)

idxs = length(Xs)
best_loss = vloss
last_improvement = epoch
epoch_start_time = now() 
epochs += epoch
epoch += 1
for epoch_idx in epoch:epochs
  global best_loss,last_improvement,t_sec,epoch_start_time
  mean_loss = 0.0f0
  for (idx,(xs,ys)) in enumerate(zip(Xs, Ys))
    Flux.train!(loss, params(m), [(xs|>gpu,ys|>gpu)], opt)
    lss = loss(tx,ty)
    mean_loss += lss
    if idx % 10 == 0
      @info "epoch# $(epoch_idx)/$(epochs)-$(idx)/$(idxs) loss = $(lss)";flush(log)
    end
  end
  mean_loss /= idxs

  run_sec = round(Millisecond(now()-epoch_start_time), Second)
  run_min = round(Second(run_sec), Minute)
  t_run_min = round(Second(t_sec+run_sec), Minute)
  @info "epoch# $(epoch_idx)/$(epochs)-> mean loss : $(mean_loss), running time : $(run_min)/$(t_run_min)";flush(log)
  
  # If this is the best accuracy we've seen so far, save the model out
  if mean_loss <= best_loss
    @info " -> New best loss! saving model out to $(model_file)"; flush(log)
    model = m |> cpu
    vloss = mean_loss;epoch = epoch_idx; sec = t_sec + run_sec
    # @save,@load 시 같은 이름을 사용해야 함, 여기서는 "model"을 사용함
    @save model_file model vloss epoch sec
    best_loss = mean_loss
    last_improvement = epoch_idx    
  end

  # If we haven't seen improvement in 5 epochs, drop out learning rate:
  if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
    opt.eta /= 10.0
    @info " -> Haven't improved in a while, dropping learning rate to $(opt.eta)!";flush(log)
    # After dropping learning rate, give it a  few epochs to improve
    last_improvement = epoch_idx
  end  

  if epoch_idx - last_improvement >= 10  
    @info " -> We're calling this converged."; flush(log)
    break
  end  
end
end_time = now()
@info "End - $(end_time)";flush(log)
run_min = round(round(Millisecond(end_time - start_time), Second),Minute)
@info "Running time : $(run_min)";flush(log)

# Sampling

function sample(m, alphabet, len)
  m = cpu(m)
  Flux.reset!(m)
  buf = IOBuffer()
  c = rand(alphabet)
  for i = 1:len
    write(buf, c)
    c = wsample(alphabet, m(onehot(c, alphabet)))
  end
  return String(take!(buf))
end

@info sample(m, alphabet, 1000);flush(log)

if create_log_file
  close(log)
end

