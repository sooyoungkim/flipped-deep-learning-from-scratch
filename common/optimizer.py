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

        for key in params.keys():
            # Vt = mu * Vt-1 - lr*Gt
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            # Wt+1 = Wt - lr*G*(1+mu) + mu(mu*Vt-1)
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                # h 초기화
                self.h[key] = np.zeros_like(val)
                # 예 1) native
                # for 1회 -> key: x , val: -7.0 -> self.h = {'x': array(0.)}
                # for 2회 -> key: y , val:  2.0 -> self.h = {'x': array(0.), 'y': array(0.)}

                # 예 2) mnist
                # for 1회 ->
                # {'W1': array([[0., 0., 0., ..., 0., 0., 0.],
                #        ...,
                #        [0., 0., 0., ..., 0., 0., 0.]])}
                # for 10회 ->
                # {
                #  'W1': array([[0., 0., 0., ..., 0., 0., 0.],
                #        ...,
                #        [0., 0., 0., ..., 0., 0., 0.]])
                # , 'b1': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                #        ...,
                #        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}
                # ...,
                # , 'W5': array([[0., 0., 0., ..., 0., 0., 0.],
                #        ...,
                #        [0., 0., 0., ..., 0., 0., 0.]])
                # , 'b5': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                #        ...,
                #        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}
                # }

        # params key(x와 y가 있다) 별로 반복
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
            params[key] -= self.lr * grads[key] / ((np.sqrt(self.h[key])) + 1e-8)

            # 예 1)
            # for 1회 ->
            # self.h[x] = 변경전 : 0.0 -> (기울기 :  -0.7) -> 변경 후 :  0.48999999999999994
            # self.h = {'x': array(0.49), 'y': array(0.)}
            # params = {'x': -5.5, 'y': 2.0}
            # self.h[x] = 변경전 : 0.0 -> (기울기 :   4.0) -> 변경 후 :  16.0
            # self.h = {'x': array(0.49), 'y': array(16.)}
            # params = {'x': -5.5, 'y': 0.5}
            # for 2회 ->
            # self.h[x] = 변경전 : 0.48999999999999994 -> (기울기 :  -0.55) -> 변경 후 : 0.7925
            # self.h = {'x': array(0.7925), 'y': array(16.)}
            # params = {'x': -4.573267672102144, 'y': 0.5}
            # self.h[x] = 변경전 : 16.0 -> (기울기 :   1.0) -> 변경 후 : 17.0
            # self.h = {'x': array(0.7925), 'y': array(17.)}
            # params = {'x': -4.573267672102144, 'y': 0.13619656244550055}

            # 예 2)
            # sqrt :
            # [[0. 0. 0. ... 0. 0. 0.]
            #  [0. 0. 0. ... 0. 0. 0.]
            #  ...
            #  [0. 0. 0. ... 0. 0. 0.]
            #  [0. 0. 0. ... 0. 0. 0.]]
            # sqrt :
            # [0.04866351 0.07532173 0.01063748 0.00808339 0.01587566 0.01184867
            #    0.0171543  0.01997456 0.0080663  0.00424146 0.0576386  0.0115372
            #    0.02674752 0.00071907 0.1010608  0.02649115 0.03244322 0.03792477
            #    0.00488895 0.01355946 0.00056699 0.0144815  0.06423799 0.01170992
            #    0.00327724 0.01005131 0.01021535 0.00909711 0.00234215 0.01640358
            #    0.03622145 0.01044305 0.00017293 0.00350577 0.0122549  0.00256546
            #    0.03470251 0.0127562  0.02260998 0.00318505 0.01246058 0.01353869
            #    0.00465887 0.09027203 0.00518771 0.00508613 0.00072709 0.00780842
            #    0.00441049 0.00600924 0.00519525 0.0035017  0.00978354 0.00847481
            #    0.05078916 0.0025433  0.00492335 0.02007352 0.01286765 0.01759862
            #    0.01877192 0.02800164 0.09240301 0.04325294 0.05132428 0.00310049
            #    0.01531935 0.01293895 0.02675745 0.0245683  0.01167872 0.04940813
            #    0.00835613 0.00610707 0.00175341 0.00620666 0.00960615 0.02725865
            #    0.0418529  0.00116712 0.00233155 0.02897166 0.0079622  0.00462554
            #    0.009523   0.00201789 0.00430852 0.0091255  0.00049532 0.00344851
            #    0.04186317 0.01335907 0.00057829 0.00106896 0.00162381 0.00343208
            #    0.01040481 0.04373075 0.02622401 0.0149598]


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

