import numpy as np
from matplotlib import pyplot as plt

# load data
N = 100
X = np.random.rand(N, 1)
Y = X + np.random.normal(0, 0.1, (N, 1))

# adaline
w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)) , X.T) , Y)

# show result
Y_hat = np.matmul(X , w)
plt.scatter(X[:, 0], Y, c='red', s=10)
plt.plot(X[:, 0], Y_hat, c='blue')
plt.show()