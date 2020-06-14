import numpy as np
from matplotlib import pyplot

# load data
N = 50
X = np.random.uniform(low=0, high=20, size=(N, 1))
Y = X + np.random.normal(0, 1, (N, 1)) + 5

_, ax1 = pyplot.subplots()

# hyper parameters
epochs = 50
lr_w = 0.0001
lr_b = 0.01

w = np.random.rand(1 ,1)
b = np.random.rand(1 ,1)

for epoch in range(epochs):
    for i in range(N):
        
        # perceptron
        Y_hat = np.matmul(X[i] , w) + b
        e = Y[i] - Y_hat

        # update
        w = w + lr_w * e * X[i]
        b = b + lr_b * e

        # show result
        Y_hat = np.matmul(X , w) + b

        ax1.clear()
        ax1.scatter(X, Y, s=1, c='red')
        ax1.plot(X, Y_hat, c='blue', lw=1)
        pyplot.pause(0.01)