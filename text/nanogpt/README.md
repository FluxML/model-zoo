# Generative pre-trained transformer

![GPT architecture](docs/Full_GPT_architecture.svg)

[Source](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer)

## Model Information

GPT is built of a multi-head attention architecture.  We offer here a very small instance based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).  The default parameters give a model much smaller than nanoGPT, tuned for fastest convergence on a very small data set (Shakespeare).

This model takes as input a sequence of existing text (context) and produces as output the predicted next character.  Actually, it produces the predicted next character for each initial sub-sequence of the input, in effect giving an extra degree of parallelism for the purposes of training.

For the attention mechanism, we use [Flux.MultiHeadAttention](https://fluxml.ai/Flux.jl/stable/reference/models/layers/#MultiHeadAttention).


## Training

```shell
cd text/gpt
julia --project gpt.jl
```

## References

* [Attention is all you need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
* [Youtube (3blue1brown): Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc)
* [Youtube (Karpathy): Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
* [nanoGPT](https://github.com/karpathy/nanoGPT)
