##############################################################################
# 5.7 오차역전파법 구현하기
##############################################################################
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common import layers
from common import gradient2 as grad
# from common import gradient

from collections import OrderedDict

class TwoLayerNet:
    # 입력층 뉴런수, 은닉층 뉴런수, 출력층 뉴런수
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        # print(self.params)

        # 계층 생성
        self.layers = OrderedDict()  # 신경망 계층을 OrderedDict(순서가 있는 딕셔너리)에 보관하는 점이 중요
        self.layers['Affine1'] = layers.Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = layers.Relu()
        self.layers['Affine2'] = layers.Affine(self.params['W2'], self.params['b2'])
        # 마지막 출력 층 생성
        self.lastLayer = layers.SoftmaxWithLoss()

    # 추론 : 예측하기
    def predict(self, x):
        # 추가한 순서대로 각 계층의 forward() 메서드를 호출하기만 하면 처리된다.
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # 손실함수, 비용함수
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    # x: 입력 데이터, t: 정답 레이블 (원-핫 인코딩 형태가 아니다)
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)    # 행단위로 가장 큰 값이 있는 위치의 index를 반환

        if t.ndim != 1 :            # 다차원 배열이면(1차원 벡터가 아니면)
            t = np.argmax(t, axis=1)

        # 정확도 계산 = 정답인 개수 / 전체 데이터 개수
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

        # 기울기 구하기 & 업데이트 (방법1)
        # x : 입력 데이터, t : 정답 레이블

    def numerical_gradient(self, x, t):
        print(x.shape)  # (batch_size, 784)
        print(t.shape)  # (batch_size, 10)
        # print(self.params['W1'])
        # print("=========")
        # print(x)
        # print("=========")
        # print(self.params['W2'])
        # print("=========")
        # print(x)
        # print("=========")

        # 입력데이터, 정답레이블은 동일하게 유지되면서 매개변수만 갱신된다.
        loss_W = lambda W: self.loss(x, t)  # W는 더미 용도(사용하지 않음) -> 기울기 계산에서 f(x) 호출하는 형태를 맞춰주기 위함

        # 손실 함수의 값을 줄이기위해 각 가중치, 편향 매개변수의 기울기를 구해서 업데이트한다.
        grads = {}
        grads['W1'] = grad.numerical_gradient(loss_W, self.params['W1'])  # loss_W(self.params['W1']) 미분
        grads['b1'] = grad.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = grad.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = grad.numerical_gradient(loss_W, self.params['b2'])

        # 동일할까? 다르다!!
        #print("grads['W1'] : ", grads['W1'])            # grads로 업데이트 하게된다.
        #print("self.params['W1'] : ", self.params['W1'])

        return grads


    # 기울기 구하기 & 업데이트
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        # 반대 순서로 각 계층의 backward() 메서드를 호출하기만 하면 처리된다
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads



if __name__ == '__main__':
    t = np.array([1, 0, 0])
    print(t.ndim)   # 1
    t = np.array([[1, 0, 0], [0, 1, 0]])
    print(t.ndim)   # 2

    print(t)
    # [[1 0 0]   -> 0번째 인덱스 값이 가장 크다.
    #  [0 1 0]]  -> 1번째 인덱스 값이 가장 크다.
    print(np.argmax(t, axis=1))   # [0 1]

    t1 = np.array([[4, 5, 6], [2, 7, 14]])
    print(t1.shape[0])  # 2
    print(t1 / t1.shape[0])
    # [[2.  2.5 3.]
    #  [1.  3.5 7.]]
