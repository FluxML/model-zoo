# ELM  
## Julia implementation of ELM  
[ELM](http://www.ntu.edu.sg/home/egbhuang/) (Extreme Learning Machine) is **S**ingle **L**ayer **F**eedforward **N**eural Network.  
Wikipedia: https://en.wikipedia.org/wiki/Extreme_learning_machine  
It uses [random projections](https://en.wikipedia.org/wiki/Random_projection) and [least-squares fit](https://en.wikipedia.org/wiki/Least-squares_fit) to learn parameters.  

### Folder Contents:  

#### ELM.jl  
> Julia file that implements Extreme Learning Machine algorithm to learn and predict Flux MNIST dataset of which 80% samples were used for training and 20% samples were used as validation set.  


### Results obtained:

| Dataset       |Number of hidden nodes | Accuracy      | Training Time (i5 8250) |
|:-------------:|:---------------------:|:-------------:|:-----------------------:|
| Training      | 1000                  |  94%          |	4 sec 				  |
| Validation    | 1000                  |  94%          |						  |
| Training      | 6500                  |  99%          |	58 sec				  |
| Validation    | 6500                  |  98%          |	-					  |
