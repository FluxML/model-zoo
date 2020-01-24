# # Character-level Recurrent Neural Network
#- Train model on Shakespeare's works
#- Have model write like Shakespeare at the end

# # 1. Import Dependencies

using Flux
using Flux: onehot, chunk, batchseq, throttle, crossentropy
using StatsBase: wsample
using Base.Iterators: partition

# # 2. Data Download & Pre-processing
# - Source of data: Shakespeare text from https://cs.stanford.edu/people/karpathy/char-rnn/
# - Generate character tokens
# - Partition in batches for input
cd(@__DIR__)

isfile("input.txt") ||
  download("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
           "input.txt")

#Generate array of all chars appearing in input.txt, let total num be N:
text = collect(String(read("input.txt")))
alphabet = [unique(text)..., '_'] #get unique char array
#Generate array of one-hot vectors for each character in the text. 
#Each vector has N-elements, where 1 element in N is set to true (others: false):
text = map(ch -> onehot(ch, alphabet), text)
stop = onehot('_', alphabet) #generate end token

N = length(alphabet)
seqlen = 50 #batch size
nbatch = 50 #number of batches

# perform chunking to get meaningful phrases, partition into minibatches and return as array
Xs = collect(partition(batchseq(chunk(text, nbatch), stop), seqlen))
Ys = collect(partition(batchseq(chunk(text[2:end], nbatch), stop), seqlen))

# # 3. Define RNN Model, Hyperparameters
#- Define 4-layer deep RNN
#- Define loss function as Cross Entropy loss
#- Define optimiser as Adam with learning rate of 0.01
#Flux's chain function joins multiple layers together, such that layer operations are performed on input sequentially. 
m = Chain(
  LSTM(N, 128), #Long Short-term Memory of feature space size 128
  LSTM(128, 128), # output is 128-dimensional
  Dense(128, N), #N = number of possible tokens
  softmax) #calculate the probability of output char corr. to each possible char

m = gpu(m) #use GPU acceleration

function loss(xs, ys) #CE loss, or log loss quanitfies the performance of models with probability output
  l = sum(crossentropy.(m.(gpu.(xs)), gpu.(ys))) #pass to GPU and get cost
  Flux.truncate!(m)
  return l
end

opt = ADAM(0.01) #use the ADAM optimiser with learning rate of 0.01
tx, ty = (Xs[5], Ys[5])
evalcb = () -> @show loss(tx, ty)

# # 4. Train model
Flux.train!(loss, params(m), zip(Xs, Ys), opt,
            cb = throttle(evalcb, 30)) #timeout for 30 secs

# # 5. Sample from input.txt and test model
# Compose a 1000-char long verse in the style of Shakespeare!
function sample(m, alphabet, len)
  m = cpu(m) #use cpu as gpu offers minimal acc for seq models
  Flux.reset!(m)
  buf = IOBuffer()
  c = rand(alphabet) #take random input char token
  for i = 1:len
    write(buf, c)
    #Compose like Shakespeare char-by-char! :
    c = wsample(alphabet, m(onehot(c, alphabet)).data)
  end
  return String(take!(buf)) #get results from last LSTM hidden state
end

#Print results
sample(m, alphabet, 1000) |> println
