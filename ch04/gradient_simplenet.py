##############################################################################
# 4.4.2 신경망에서의 기울기
#   가중치 매개변수에 대한 손실한수의 기울기를 구한다.
#   가중치 W를 조금 변경했을때 손실함수 L이 얼마나 변화하느냐를 나타낸다.
##############################################################################
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common import functions
from ch04 import gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = functions.softmax(z)
        loss = functions.cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
dW = gradient.numerical_gradient(f, net.W)

print(dW)
