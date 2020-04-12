import numpy as np
from matplotlib import pyplot as plt

# load data
N = 100
X = np.random.rand(N, 1)
Y = X + np.random.normal(0, 0.1, (N, 1))

_, ax1 = plt.subplots()

ERRORS = np.array([])

# hyper parameters
epochs = 10
lr = 0.05

w = np.random.rand(1 ,1)

for epoch in range(epochs):
    for i in range(N):
        
        # perceptron
        Y_hat = np.matmul(X[i] , w)
        e = Y[i] - Y_hat
        ERRORS = np.append(ERRORS, e)

        # update
        w = w + lr * e * X[i].T

        # show result
        Y_hat = np.matmul(X , w)
        
        ax1.clear()
        ax1.scatter(X, Y, s=1, c='red')
        ax1.plot(X, Y_hat, c='blue', lw=1)
        plt.pause(0.01)

_, ax2 = plt.subplots()
ax2.plot(ERRORS)
plt.show()