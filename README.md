# bruteforce

### Purpose
Brute force scanning of parameters of neural networks for regression problems.

### Motivation
Optimal parameters of neural networks can be difficult to estimate from theoretical considerations. It is mathematically proven that neural networks work effectively for many regression problems.
However, instructions for finding optimal network structures from theoretical considerations are often not available. Besides, more than one optimal network structure ensures a sufficient model performance. 

Therefore an automatic configuration of network structures is desired. This covers variations of the number and size of hidden layers, the activation functions of hidden and output layer, 
parameters of early stopping of the training or of deacy of weights, and the effect of random initialization of the weights.   

### Options
- Brute force scanning of network parameter space (slow, but transparent) 
- Automatic solutions such as Google’s AutoML (automatic regulariztion, but closed hood with the risk of insufficient model understanding)

Due to its explicit transparency and robust implementation, brute force scanning has been implemented. The exhaustive search relies merely on guessing wide parameter ranges and eliminates the risk of getting trapped in local optima.

### Implementation
bruteforce.py provides nested search loops over selected parameter ranges. The choice of the best configuration is based on the mean squared error, see metrics.py.
BruteForce uses different backends (e.g. TensorFlow, NeuroLab). The library specific functions are implemented in the children of class BruteForce:
- Class NeuralTf: Tensorflow/Keras variant
- Class NeuralNl: Neurolab variant
BruteForce is only dependent on the files stored in the code directory of this repository.

### Example
test_bruteforce.py is an example for using the backends TensorFlow and NeuroLab for a simple regression problem in 1D space.  

### Dependencies
- Module _neuralnl_ is dependent on package _neurolab_ [[NLB15]](https://github.com/dwweiss/grayboxes/wiki/References#nlb15)
- Module _neuraltf_ is dependent on package _tensorflow_ [[ABA15]](https://github.com/dwweiss/grayboxes/wiki/References#aba15)

Installation of the packages needed can be done with: 
    pip install tensorflow neurolab matplotlib numpy

