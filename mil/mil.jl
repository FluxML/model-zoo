# implements the multiple-instance learning model using Neural Networks, as described in
# https://arxiv.org/abs/1609.07257
# Using Neural Network Formalism to Solve Multiple-Instance Problems, Tomas Pevny, Petr Somol

using FileIO
using Flux
using Base.Iterators: repeated
using Flux: throttle
import Flux.Optimise: ADAM

# determines boundaries of each sample from instance identifiers in the form [1,1,1,2,2,3,3,3,3,....,n,n,n]
function findranges(ids::AbstractArray)
  bags=Vector{UnitRange{Int64}}()
  idx=1
  for i in 2:length(ids)
    if ids[i]!=ids[idx]
      push!(bags,idx:i-1)
      idx=i;
    end
  end
  if idx!=length(bags)
    push!(bags,idx:length(ids))
  end
  return(bags)
end

# load the musk dataset
function loaddata()
  data = readdlm("musk.csv",',');
  y = Int.(data[:,1])
  bags = findranges(data[:,2])
  x = data[:,3:end]'
  y = map(b -> maximum(y[b]),bags) + 1
  return(y,bags,x)
end

# calculate average over instances of individual samples implemented by product with a sparse matrix
function segmented_mean(x,segments::Vector{UnitRange{Int64}})
  rowval = collect(1:size(x,2))
  nzval = zeros(eltype(x),size(x,2))
  colptr = zeros(Int,length(segments)+1)
  colptr[end] = size(x,2)+1
  for (i,segment) in enumerate(segments)
      colptr[i] = segment.start
      nzval[segment] = 1./length(segment)
  end
  m = size(x,2)
  n = length(segments)
  x*SparseMatrixCSC(m,n,colptr,rowval,nzval)
end

# implements the Multiple-instance learning model, where pre / post identifies the processing before / after the aggreagation
struct BagModel{A,F,B}
    pre::A
    aggregation::F
    post::B
end

# forward pass for the flux model 
(m::BagModel)(x,bags) = m.post(m.aggregation(m.pre(x),bags))
Flux.Optimise.children(m::BagModel) = [m.pre,m.post]


# load the data, define the model and loss function
(y,bags,x) = loaddata();
model = BagModel(Dense(size(x,1),10,Flux.relu),
  segmented_mean,
  Chain(Dense(10,10,Flux.relu),Dense(10,2)));
loss(x,bags,y) = Flux.crossentropy(model(x,bags),y);

# initialize the training algorithm and iterations
dataset = repeated((x,bags, y), 20000)
evalcb = () -> @show(loss(x, bags, y))
opt = ADAM(params(model))

# train
Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))

# calculate the error on the training set (no testing set right now)
mean(mapslices(indmax,model(x,bags).data,1)' .!= y)
