"""
  Copyright (c) 2016- by Dietmar W Weiss

  This is free software; you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as
  published by the Free Software Foundation; either version 3.0 of
  the License, or (at your option) any later version.

  This software is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this software; if not, write to the Free
  Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
  02110-1301 USA, or see the FSF site: http://www.fsf.org.

  Version:
      2021-01-15 DWW
"""

import os
import sys
sys.path.append(os.path.abspath('..'))  # modules are located in '../grayboxes'
sys.path.append(os.path.abspath('.'))   # modules are located in this directory

import matplotlib.pyplot as plt
import numpy as np
import unittest

try:
    from grayboxes.neuraltf import Neural as NeuralTf
except:
    from neuraltf import Neural as NeuralTf
try:
    from grayboxes.neuralnl import Neural as NeuralNl
except:
    from neuralnl import Neural as NeuralNl


class TestUM(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1(self):
        N = 1000                               # number of training sets
        n = int(1.4 * N)                           # number of test sets
        nse = 5e-2                           # noise relative to x-value
        
        X = np.linspace(-2. * np.pi, 2. * np.pi, N).reshape(-1, 1)
        dx = 0.25 * (X.max() - X.min())
        x = np.linspace(X.min() - dx, X.max() + dx, n).reshape(-1, 1)
        Y_tru = np.sin(X)
        Y = Y_tru + np.random.uniform(-nse, +nse, size=X.shape)
        y_tru = np.sin(x)
        
        plt.title('train data & true values')
        plt.plot(X, Y, label='train')
        plt.plot(x, y_tru, ':', label='true')
        plt.legend(); plt.grid(); plt.show()
        
        for backend in [
#                NeuralNl, 
                NeuralTf,
                ]:
            phi = backend()
            y = phi(X=X, Y=Y,                # training input and target 
                    x=x,                              # prediction input
                activation=('leaky',)  # 'elu',) 
                    if phi.backend == 'tensorflow' else 'sigmoid',
                epochs=150,
                expected=1e-3 if phi.backend == 'tensorflow' else 1e-3,
                                                          # expected MSE
                learning_rate=0.1,            # tensorflow learning rate
                neurons=[[i]*j for i in range(4, 4+1)       # i: neurons  
                               for j in range(2, 2+1)],       # j: layer
                output='linear',   # activation function of output layer
                patience=10,           # tensorflow early stopping delay
                plot=1,           # 0: none, 1: final only, 2: all plots 
                rr=0.1,                   # neurolab:bfgs regularization
                show=1,                        # neurolab plot frequence
                tolerated=5e-3,                          # tolerated MSE
                trainer='adam' if phi.backend == 'tensorflow' else 'bfgs',
                trials=5,   # repetition of every training configuration 
                )
            
            if y is None:
                print('??? y is None -> training failed')
            
        plt.title('train and prediction data')
        y = y[:, 0]           # children of BruteForce return a 2D array 
        plt.plot(X, Y, label='train')
        plt.plot(x, y_tru, ':', label='true')
        plt.plot(x, y, label='pred')
        plt.legend(); plt.grid(); plt.show()

        self.assertIsNotNone(y)


if __name__ == '__main__':
    unittest.main()
