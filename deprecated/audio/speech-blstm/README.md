This model is an implementation of the neural network for speech recognition described in Graves & Schmidhuber (2005). It takes in frames of frequency information derived from the waveform, and it predicts which phone class the frame belongs to, among a reduced set of English phones. The training is run using the [TIMIT data set (Garofolo et al., 1993)](https://catalog.ldc.upenn.edu/LDC93S1).

# How to use these scripts

This implementation is broken down into two separate scripts. The first, `00-data.jl`, extracts the appropriate speech features from the data in TIMIT and saves them to file. It assumes that you have the TIMIT speech corpus extracted, [converted into RIFF WAV file format](https://web.archive.org/web/20180528013655/https://stackoverflow.com/questions/47370167/change-huge-amount-of-data-from-nist-to-riff-wav-file), and in the same directory as the script itself. It takes no arguments, and is run

```bash
julia 00-data.jl
```

It will print out which directory it is working on as it goes so you can track the progress as it extracts the training and testing data.

The second script, `01-speech-blstm.jl`, trains the network. It loads in the speech data extracted from `00-data.jl` and runs it through the network for 20 epochs, which is on average how long Graves & Schmidhuber needed to train the network for. (The number of epochs can be changed by modifying the value of the `EPOCHS` variable in the script.) The script is run as

```bash
julia 01-speech-blstm.jl
```

At the end of each epoch, the script prints out the validation accuracy and saves a BSON file with the model's current weights. After running through all the epochs, the script prints out the testing accuracy on the default holdout test set.

# Using a trained model

It is simple to use the model once it's been trained. Simply load in the model from the BSON file, and use the `model(x)` function from `01-speech-blstm.jl` on some data prepared using the same procedure as in `00-data.jl`. The phoneme class numbers can be determined by using `argmax`. The `Flux` and `BSON` packages will need to be loaded in beforehand.

```julia
using Flux, BSON
using Flux: flip, softmax
BSON.@load "model_epoch20.bson" forward backward output
BLSTM(x) = vcat.(forward.(x), flip(backward, x))
model(x) = softmax.(output.(BLSTM(x)))
ŷ = model(x) # where x is utterance you want to be transcribed
phonemes = argmax.(ŷ)
```

# References

Garofalo, J. S., Lamel, L. F., Fisher, W. M., Fiscus, J. G., Pallett, D. S., & Dahlgren, N. L. (1993). The DARPA TIMIT acoustic-phonetic continuous speech corpus cdrom. Linguistic Data Consortium.

Graves, A., & Schmidhuber, J. (2005). Framewise phoneme classification with bidirectional LSTM and other neural network architectures. *Neural Networks, 18*(5-6), 602-610.
