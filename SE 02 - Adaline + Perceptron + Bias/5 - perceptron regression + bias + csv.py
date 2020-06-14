import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def MeanSquaredError(X, Y, N, w, b):
    sum = 0
    for j in range(N):
        Y_hat = np.matmul(X[j, :], w) + b
        error = Y[j] - np.sign(Y_hat)
        sum += (error ** 2)

    return sum / N

# load data
data = np.array(pd.read_csv('linear_data_train.csv', skiprows=1, header=None))

X = np.array(data[:, 0:2])
Y = np.array(data[:, 2])

N = X.shape[0]

ERROR = np.array([])

_, ax1 = plt.subplots()

# hyper parameters
epochs = 2
lr = 0.01

w = np.random.rand(2 ,1)
b = np.random.rand(1 ,1)

for epoch in range(epochs):
    for i in range(N):
        
        # perceptron
        Y_hat = np.matmul(X[i] , w) + b
        
        e = Y[i] - Y_hat
        mse = MeanSquaredError(X, Y, N, w, b)
        ERROR = np.append(ERROR, mse)

        # update
        w = w + lr * e * X[i].T
        b = b + lr * e

        # show result
        Y_hat = np.matmul(X , w) + b

        ax1.clear()
        ax1.plot(ERROR)
        plt.pause(0.01)
       