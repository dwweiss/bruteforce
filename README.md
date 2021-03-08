# bruteforce  

### Purpose
Brute force scanning of configurations of neural network for regression tasks.

### Motivation
Optimal parameters of neural networks can be difficult to estimate from theoretical considerations. It is mathematically proven that neural networks work effectively for most regression problems of higher complexity.
However, algorithmic instructions for finding the optimal network configuration are often not available. Besides, more than one optimal network structure ensure sufficient model performance. 

Therefore an automatic configuration of network parameters is proposed. This covers variations of the number and size of hidden layers, the activation functions of hidden and output layers, parameters of early stopping of the training or of deacy of weights, the effect of random initialization of the network weights etc.   

### Options for finding the optimal configuration
- Brute force scanning of network parameter space (slow, but transparent) 
- Automatic solutions such as Google’s AutoML (automatic regulariztion, but closed hood with the risk of insufficient model understanding)

Due to its explicit transparency and robust implementation, brute force scanning has been implemented. The exhaustive search relies merely on guessing wide parameter ranges and eliminates the risk of getting trapped in local optima.

### Implementation
Class _BruteForce_ in module _bruteforce_ performes nested search loops over selected parameter ranges. 

![loops](https://github.com/dwweiss/bruteforce/blob/main/bruteforce/doc/fig/brute_force_loops.PNG)

###### Figure 1: Loops (MSE: mean squared error)

The choice of the best configuration is based on the mean squared error, see module _metrics_.  _BruteForce_ uses different backends (e.g. TensorFlow, NeuroLab). Library specific functionality is implemented in the children of class _BruteForce_:
- Class _NeuralTf_: Tensorflow/Keras variant
- Class _NeuralNl_: Neurolab variant

### Example: Sine curve
_test_bruteforce.py_ is an example using synthetic data in 1D space with the backends TensorFlow and NeuroLab.  

        N = 1000                    # number of training sets
        n = int(np.round(1.4 * N))  # number of test sets
        nse = 5e-2                  # noise
        
        X = np.linspace(-2. * np.pi, 2. * np.pi, N).reshape(-1, 1)
        dx = 0.25 * (X.max() - X.min())
        x = np.linspace(X.min() - dx, X.max() + dx, n).reshape(-1, 1)
        Y_tru = np.sin(X)
        Y = Y_tru + np.random.uniform(-nse, +nse, size=X.shape)
        y_tru = np.sin(x)
                
        for backend in [
                #  NeuralNl, 
                NeuralTf,
                ]:
            phi = backend()
            y = phi(X=X, Y=Y, x=x,
                activation=('leaky', 'elu',) 
                    if phi.backend == 'tensorflow' else 'sigmoid',
                epochs=150,
                expected=1e-3 if phi.backend == 'tensorflow' else 1e-3,
                learning_rate=0.1,            # tensorflow learning rate
                neurons=[[i]*j for i in range(4, 4+1)       # i: neurons  
                               for j in range(4, 4+1)],       # j: layer
                output='linear',
                patience=10,      # delay of tensorflow's early stopping
                plot=1,           # 0: none, 1: final only, 2: all plots 
                rr=0.1,                   # neurolab:bfgs regularization
                show=1,
                tolerated=5e-3,
                trainer='adam' if phi.backend == 'tensorflow' else 'bfgs',
                trials=5,   # repetition of every training configuration 
                )

#### Results

The training data and the true values are plotted in Figure 2.

![train_and_true](https://github.com/dwweiss/bruteforce/blob/main/bruteforce/doc/fig/bruteforce_train_and_true1.png)

###### Figure 2: Training data and true values without noise


Figure 3 shows the history of the mean squared error of all trials for the TensorFlow backend. 

![history_all](https://github.com/dwweiss/bruteforce/blob/main/bruteforce/doc/fig/bruteforce_history1_all.png)

###### Figure 3: Mean squared error history of all trials


In Figure 4 the history of the five best trials out of all trials plotted in Figure 3 is shown. 

![history_5best](https://github.com/dwweiss/bruteforce/blob/main/bruteforce/doc/fig/bruteforce_history1_5best.png)

###### Figure 4: Mean squared error history of five best trials


The resulting errorbars are summarized in Figure 5. 

![MSE_history_all](https://github.com/dwweiss/bruteforce/blob/main/bruteforce/doc/fig/bruteforce_errorbars1.png)

###### Figure 5: Errorbars of all trials

It is obvious that a single training is risky, see MSE of training with _leakyrelu_ in Figure 5. The first trial (#0) fails perfectly. Therefore a required minimum number of 3 repetitions is advised.



### Example: UIC airfoil + noise dataset

This real-world example with 6 input, 1 output and 1503 data points is taken from the UIC database:

https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise

Each of the 5 hidden layers contain 8 neurons. The trainer is _adam_, the types of activation of hidden layers were: (_elu_, _leakyrelu_, _sigmoid_) and every configuration was repeated 5 times.   

Figure 6 shows the history of the mean squared error of all trials for the TensorFlow backend. 

![history_all](https://github.com/dwweiss/bruteforce/blob/main/bruteforce/doc/fig/bruteforce_history_uic_airfoil.png)

###### Figure 6: Mean squared error history of all trials


The resulting errorbars are summarized in Figure 7. 

![MSE_history_all](https://github.com/dwweiss/bruteforce/blob/main/bruteforce/doc/fig/bruteforce_errorbars_uic_airfoil.png)

###### Figure 7: Errorbars of all trials

A single training is risky, see MSE of training with hidden neuron activation _sigmoid_ in Figure 6. The first trial (#0) with _sigmoid_ fails. The influence of the choice of the activation function is little, but here is an indication that the MSE variation with activation _leakyrelu_ is less than the variation with the other activation functions. In contrast, the sine curve example has indicated that _leakyrelu_ is not a good choice. The recommended minimum number of repetitions is 3. 


### Conclusion

The required number of training repetitions is highly problem-specific in regression analysis of measurements. There are both examples were a single training is sufficient and examples were multiple training trials are definitely needed. 

Single training of a network configuration on a new dataset can represent a substantial risk of missing an acceptable solution. A preference of a particular trainer or activation of the hidden layers has not been found. Brute force scanning of the network parameters is therefore recommended.

The conclusion is the need of 3-5 repetitions of the random initialization of weights for each network configuration in case of new data sets. 

### Installation

#### Dependencies
- Module _neuralnl_ is dependent on package _neurolab_ [[NLB15]](https://github.com/dwweiss/grayboxes/wiki/References#nlb15)
- Module _neuraltf_ is dependent on package _tensorflow_ [[ABA15]](https://github.com/dwweiss/grayboxes/wiki/References#aba15)

One way of installation of the needed packages is: 

    pip install tensorflow=2.2.2 neurolab matplotlib numpy pandas scipy

#### Run test

If the downloaded zip-file _bruteforce-main.zip_ is extracted in a local directory, the example file _test_bruteforce.py_ can be directly excuted in the _bruteforce_ sub-directory:

    bruteforce-main > bruteforce > python test_bruteforce.py

