using Flux
using Flux: onehot, chunk, batchseq, throttle, crossentropy
using StatsBase: wsample
using Base.Iterators: partition
using Parameters: @with_kw
using ProgressMeter

cd(@__DIR__)

@with_kw struct HyperParams
    nbatch::Int = 50
    seqlen::Int = 50
    epochs::Int = 1
    lr::Float64 = 0.01
    val_char_len::Int = 1000
    verbose_freq::Int = 1
end

function get_data(hparams::HyperParams)    
    isfile("input.txt") ||
      download("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
               "input.txt")
    
    text = collect(String(read("input.txt")))
    alphabet = [unique(text)..., '_']
    text = map(ch -> onehot(ch, alphabet), text)
    stop = onehot('_', alphabet)
    N = length(alphabet)

    Xs = collect(partition(batchseq(chunk(text, hparams.nbatch)|>gpu, stop), hparams.seqlen))
    Ys = collect(partition(batchseq(chunk(text[2:end], hparams.nbatch)|>gpu, stop), hparams.seqlen))

    return Xs, Ys, alphabet, N
end

function sample(m, alphabet, hparams::HyperParams)
    Flux.reset!(m)
    buf = IOBuffer()
    c = rand(alphabet)
    for i = 1:hparams.val_char_len
        write(buf, c)
        c = wsample(alphabet, m(onehot(c, alphabet)))
    end
    return String(take!(buf))
end

function train(; kws...)
    # Parameters for training
    hparams = HyperParams()
    # data
    Xs, Ys, alphabet, N = get_data(hparams)
    # model
    m = Chain(LSTM(N, 128),
              LSTM(128, 128),
              Dense(128, N),
              softmax) |> gpu
    # optimizer
    opt = ADAM(hparams.lr)
    # validation data
    tx, ty = (Xs[5], Ys[5])
    # progress bar
    p = Progress(hparams.epochs * length(Xs))

    function loss(xs, ys)
        l = sum(crossentropy.(m.(gpu.(xs)), gpu.(ys)))
        return l
    end
    
    for ep in 1:hparams.epochs
        iter = 0
        # train
        Flux.train!(params(m), zip(Xs, Ys), opt,
                    cb=function ()
                        if iter % hparams.verbose_freq == 0
                            # calc val loss 
                            val_loss = sum(crossentropy.(m.(tx), ty))
                            info_val = [(:epoch, ep), (:iter, iter), (:train_loss, loss), (:val_loss, val_loss)]
                        else
                            info_val = [(:epoch, ep), (:iter, iter), (:train_loss, loss)]
                        end
                        ProgressMeter.next!(p; showvalues=info_val)
                        iter += 1
                    end
                    ) do x, y
            # calc loss
            loss = sum(crossentropy.(m.(x), y))
        end
        # sample and dump sequence of chars
        write("output/sample_$(ep).txt", sample(m, alphabet, hparams))
    end              
end

train()
