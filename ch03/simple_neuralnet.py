import numpy as np
from common import functions


def init_network():
    # 입력층 뉴런수 : 2 -> 은닉층1의 뉴런수 : 3 -> 은닉층2의 뉴런수 : 2 -> 출력층 뉴런수 : 2 개로 구성됨을 알 수 있다.
    network = {}

    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])    # 2 X 3
    network['b1'] = np.array([0.1, 0.2, 0.3])

    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])  # 3 X 2
    network['b2'] = np.array([0.1, 0.2])

    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])              # 2 X 2
    network['b3'] = np.array([0.1, 0.2])

    return network


"""신경망이 예측한 결과값

    parameter
    ---------------------------------
    network : 가중치 매개변수 담긴 딕셔너리
    x : 입력 데이터 

    :return
    ---------------------------------
    y : 신경망이 예측한 결과값
"""
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = functions.sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = functions.sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = functions.identity_function(a3)

    return y


network = init_network()
x = np.array([1.0, 0.5])    # 입력 데이터
y = predict(network, x)
print(y)                    # [0.31682708 0.69627909]
