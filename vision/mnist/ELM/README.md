# ELM  
## Julia implementation of ELM  
[ELM](http://www.ntu.edu.sg/home/egbhuang/) (Extreme Learning Machine) is **S**ingle **L**ayer **F**eedforward **N**eural Network.  
Wikipedia: https://en.wikipedia.org/wiki/Extreme_learning_machine  
It uses [random projections](https://en.wikipedia.org/wiki/Random_projection) and [least-squares fit](https://en.wikipedia.org/wiki/Least-squares_fit) to learn parameters.  

### Folder Contents:  
1. Elm.jl
> Julia file that contains Elm module. The Elm module consists of two functions, *elmtrain* and *elmpredict*.
2. ElmExample.jl
> Julia file that implements the Elm module to learn and predict Flux MNIST dataset of which 80% samples were used for training and 20% samples were used as validation set.  


### Results obtained:

| Dataset       |Number of hidden nodes | Accuracy      |
|:-------------:|:---------------------:|:-------------:|
| Training      | 1000                  |  94%          |
| Validation    | 1000                  |  94%          |
| Training      | 10000                 |  98%          |
| Validation    | 10000                 |  98%          |
