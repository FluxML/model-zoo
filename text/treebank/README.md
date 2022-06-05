# Recursive net on IMDB sentiment treebank

![treebank](../treebank/docs/treebank.png)

[Source](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)

## Model information

A recursive neural network can be used for learning tree-like structures (directed acyclic graphs). It computes compositional vector representations for prhases of variable length which are used as features for performing classification. 

This example uses the [Standford Sentiment Treebank dataset (SST)](https://nlp.stanford.edu/sentiment/index.html) which is often used as one of the benchmark datasets to test new language models. It has five different classes (very negative to very positive) and the goal is to perform sentiment analysis.


## Training

```shell
cd text/treebank
julia --project recursive.jl
```

## References

* [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)
