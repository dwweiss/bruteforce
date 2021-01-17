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
_bruteforce.py_ provides nested search loops over selected parameter ranges. The choice of the best configuration is based on the mean squared error, see metrics.py.
_BruteForce_ uses different backends (e.g. TensorFlow, NeuroLab). The library specific functions are implemented in the children of class BruteForce:
- Class _NeuralTf_: Tensorflow/Keras variant
- Class _NeuralNl_: Neurolab variant
_BruteForce_ is only dependent on the files stored in the code directory of this repository.

### Example
_test_bruteforce.py_ is an example for using the backends TensorFlow and NeuroLab for a simple regression problem in 1D space.  

        N = 1000                # number of training sets
        n = np.round(1.4 * N)  # number of test sets
        nse = 5e-2             # noise relative to x-value
        
        X = np.linspace(-2. * np.pi, 2. * np.pi, N).reshape(-1, 1)
        dx = 0.25 * (X.max() - X.min())
        x = np.linspace(X.min() - dx, X.max() + dx, n).reshape(-1, 1)
        Y_tru = np.sin(X)
        Y = Y_tru + np.random.uniform(-nse, +nse, size=X.shape)
        y_tru = np.sin(x)
        
        plt.title('train data & true values')
        plt.plot(X, Y, label='train')
        plt.plot(x, y_tru, label='true')
        plt.legend(); plt.grid(); plt.show()
        
        for backend in [
                #  NeuralNl, 
                NeuralTf,
                ]:
            phi = backend()
            y = phi(X=X, Y=Y, x=x,
                activation=('leaky', 'elu',) 
                    if phi._backend == 'tensorflow' else 'sigmoid',
                epochs=150,
                expected=1e-3 if phi._backend == 'tensorflow' else 1e-3,
                learning_rate=0.1,            # tensorflow learning rate
                neurons=[[i]*j for i in range(4, 4+1)       # i: neurons  
                               for j in range(4, 4+1)],       # j: layer
                output='linear',
                patience=10,      # delay of tensorflow's eraly stopping
                plot=1,           # 0: none, 1: final only, 2: all plots 
                rr=0.1,                   # neurolab:bfgs regularization
                show=1,
                tolerated=5e-3,
                trainer='adam' if phi._backend == 'tensorflow' else 'bfgs',
                trials=5,   # repetition of every training configuration 
                )

#### Results

The training data and the true values are plotted in Figure 1.
Figure 2 shows the history of the means squared error of all trials for the TensorFlow backend. In Figure 3 the history of the five best trials out of all trials plotted in Figure 2 is shown. The resulting errorbars are summarized in Figure 4. 

![train_and_true](https://github.com/dwweiss/bruteforce/blob/main/bruteforce/doc/fig/bruteforce_train_and_true1.png)

###### Figure 1: Training data and true values without noise

![history_all](https://github.com/dwweiss/bruteforce/blob/main/bruteforce/doc/fig/bruteforce_history1_all.png)

###### Figure 2: Mean squared error history of all trials

![history_5best](https://github.com/dwweiss/bruteforce/blob/main/bruteforce/doc/fig/bruteforce_history1_5best.png)

###### Figure 3: Mean squared error history of five best trials

![MSE_history_all](https://github.com/dwweiss/bruteforce/blob/main/bruteforce/doc/fig/bruteforce_errorbars1.png)

###### Figure 4: Errorbars of all trials


### Dependencies
- Module _neuralnl_ is dependent on package _neurolab_ [[NLB15]](https://github.com/dwweiss/grayboxes/wiki/References#nlb15)
- Module _neuraltf_ is dependent on package _tensorflow_ [[ABA15]](https://github.com/dwweiss/grayboxes/wiki/References#aba15)

One way of installation of the needed packages is: 

    pip install tensorflow neurolab matplotlib numpy

If the code of brute force is copied to a local directory, the test file _test_bruteforce.py_ can be directly excuted in that directory.  

