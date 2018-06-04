import numpy as np
import matplotlib.pylab as plt


"""활성화함수 : sigmoid

    parameter
    ----------------
    x : 입력 데이터

    return
    ----------------
    0 ~ 1 사이 값

"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


"""활성화함수 : relu

    parameter
    ----------------
    x : 입력 데이터

    return
    ----------------
    x > 0 이면 x
    나머지는    0

"""
def relu(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
z = relu(x)


plt.plot(x, y, 'r--', x, z, 'b--')
plt.ylim(-0.1, 5.1)
plt.show()


##############################################################################
# maximum() : Element-wise maximum
# Element-wise maximum.
# Compare two arrays and returns a new array containing the element-wise maxima.
##############################################################################
t = np.array([1.0, 2.0, 3.0])
print(1.0 + t)              # [2. 3. 4.]
print(1.0 / t)              # [1.         0.5        0.33333333]
# 원소별로 0과 비교해서 둘 중 큰 값을 원소로하는 새로운 배열을 만든다.
print(np.maximum(0, t))                             # [1. 2. 3.]
print(np.maximum(0, np.array([1.0, -2.0, 3.0])))    # [1. 0. 3.]
