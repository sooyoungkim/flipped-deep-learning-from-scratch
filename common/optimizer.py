import numpy as np


"""확률적 경사 하강법（Stochastic Gradient Descent）"""
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


"""모멘텀 SGD"""
class Momentum:

    """누적된 과거 gradient가 지향하고 있는 어떤 방향을
        현재 gradient에 보정하려는 방식"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                # 초기화 : 물체를 어떤 위치에서 0의 속도로 세팅하는 것과 같다.
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            # v <- av - (lr * dw)
            # 현재위치 이전까지 누적된 gradient vector : v
            #           -> gradient가 큰 흐름의 방향(가던 방향)을 지속하도록 도와준다.
            # 현재위치에서의 gradient vector : grads
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


"""NAG (Nesterov's Accelerated Gradient)
        - http://arxiv.org/abs/1212.0901
        - NAG는 모멘텀에서 한 단계 발전한 방법이다."""
class Nesterov:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        ### 추가


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            # h <- h + (dw * dw)
            # 과거의 기울기를 제곱하여 계속 더해간다.
            # 너무 작아지면 결국 거의 움직이지 않게 되고 학습을 멈춘다.
            self.h[key] += grads[key] * grads[key]
            # 1/sqrt(h)
            # 개별 매개변수에 맞춤형으로 학습률을 조정하면서 학습을 진행
            #   - 학습률 감소가 매개변수의 원소마다 다르게 적용됨을 의미한다.
            # 학습을 진행하면서 학습률을 점차 줄여가는 방법
            #   - 학습률 learning rate(step size)에 1/sqrt(h) 을 곱해 학습률을 조정
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]))


""" Root Mean Square prop
    AdaGrad 개선한 방식으로
    Adagrad의 식에서 gradient의 제곱값을 더해나가면서 구한 Gt 부분을 합이 아니라 지수평균으로 바꾸어서 대체한 방법이다.
        -> 지수이동평균(Exponential (Weighted) Moving Average, E(W)MA)
                : 과거 기울기의 반영 규모를 기하급수적으로 감소시킨다.
                  먼 과거의 기울기는 서서히 잊고 새로운 기울기 정보를 크게 반영한다.
    
    Adagrad처럼 Gt가 무한정 커지지는 않으면서 최근 변화량의 변수간 상대적인 크기 차이는 유지할 수 있다."""
class RMSprop:
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            # 분산 mean square
            # h <- ah + (1 - a) * (dw * dw)
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            # 표준편차 root mean square : 1/sqrt(h)
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-8)


"""Adam
    - http://arxiv.org/abs/1412.6980v8
    - Momentum + RMSprop 결합된 방식"""
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.m = None
        self.v = None


    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.t += 1
        alpha = self.lr * np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)

        for key in params.keys() :
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= alpha * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

