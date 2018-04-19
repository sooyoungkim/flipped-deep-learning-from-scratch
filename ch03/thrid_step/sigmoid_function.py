import numpy as np
import matplotlib.pylab as plt


t = np.array([1.0, 2.0, 3.0])
print(1.0 + t)              # [2. 3. 4.]
print(1.0 / t)              # [1.         0.5        0.33333333]
print(np.maximum(0, t))     # [1. 2. 3.]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
z = relu(x)


plt.plot(x, y, 'r--', x, z, 'b--')
plt.ylim(-0.1, 5.1)
plt.show()

