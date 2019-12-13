# Language Detection

This model uses an LSTM for character-level language detection. Given a sentence of text, each character is fed into the model and the final output determines which of five languages the sentence was written in.

First run `scrape.jl` to download a Wikipedia data set. `model.jl` contains the actual model and training code.
