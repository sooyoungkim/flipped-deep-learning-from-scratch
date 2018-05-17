##############################################################################
# 5.5 활성화 함수 계층
# 5.5.1 ReLU 계층
# 5.5.2 Sigmoid 계층
##############################################################################
import numpy as np
from common.functions import *

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        # 순전파의 출력을 인스턴스 변수 out에 보관
        out = sigmoid(x)
        self.out = out

        return out

    def backward(self, dout):
        # 순전파의 출력값을 사용하여 역전파를 구함
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None  # 가중치 매개변수의 미분
        self.db = None  # 편향 매개변수의 미분
        self.original_x_shape = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        # 입력 데이터 모양 변경(텐서 대응)
        dx = dx.reshape(*self.original_x_shape)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None    # 손실
        self.y = None       # softmax의 출력
        self.t = None       # 정답 레이블(원-핫 인코딩형태)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]     # 배치 데이터 수
        if self.t.size == self.y.size:   # 정답 레이블이 원-핫 인코딩 형태일 때
            # 원-핫 인코딩 형태이면 정답인덱스의 값만 1, 나머지는 0 이므로
            # 정답인덱스에서는 : 예측값 - 1
            # 나머지는       : 예측값 - 0 즉, 변경사항 없다.
            dx = (self.y - self.t) / batch_size     # 왜... batch_size로 나눌까??
        else:
            dx = self.y.copy()
            # 첫번째 데이터 : dx[0,정답인덱스] = dx[0,정답인덱스] - 1 & 나머지는 그대로,
            # 두번째 데이터 : dx[1,정답인덱스] = dx[1,정답인덱스] - 1 & 나머지는 그대로,
            # 세번째 데이터 : dx[2,정답인덱스] = dx[2,정답인덱스] - 1 & 나머지는 그대로,
            # ...
            dx[np.arrange(batch_size), self.t] -= 1  # zip 형태로 묶어 원-핫 인코딩 형태로 만든 후 뺄셈
            dx = dx / batch_size

        return dx


if __name__ == '__main__':
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    mask = (x <= 0)
    print(mask)
    # [[False  True]
    #  [True False]]

    test_relu = Relu()
    out = test_relu.forward(x)
    print(out)
    # [[1. 0.]
    #  [0. 3.]]

