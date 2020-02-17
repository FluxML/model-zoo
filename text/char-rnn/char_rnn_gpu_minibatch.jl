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

device!(0)
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
seqlen = 50
nbatch = 50

Xs = collect(partition(batchseq(chunk(text, nbatch), stop), seqlen))
txt = circshift(text,-1)
txt[end] = stop
Ys = collect(partition(batchseq(chunk(txt, nbatch), stop), seqlen))

model = Chain(
  LSTM(N, 128),
  LSTM(128, 128),
  Dense(128, N),
  softmax)

opt = ADAM(0.01)
m = model |>gpu

tx, ty = (Xs[2]|>gpu, Ys[2]|>gpu)

function loss2(xx, yy)
  out = 0.0f0
  for (idx, x) in enumerate(xx)
    out += crossentropy(m(x) .+ ϵ, yy[idx])
  end  
  Flux.reset!(m)
  out
end


epochs = 200
idxs = length(Xs)
for epoch_idx in 1:epochs
  for (idx,(xs,ys)) in enumerate(zip(Xs, Ys))
    Flux.train!(loss2, params(m), [(xs|>gpu,ys|>gpu)], opt)
    if idx % 10 == 0
      @info "epoch# $(epoch_idx)/$(epochs)-$(idx)/$(idxs) loss = $(loss2(tx,ty))";flush(stdout)
    end
  end
end

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

sample(m, alphabet, 1000) |> println
