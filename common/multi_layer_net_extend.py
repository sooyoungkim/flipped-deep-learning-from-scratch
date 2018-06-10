##############################################################################
#
#
##############################################################################
import sys, os
sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from collections import OrderedDict
from common import layers
from common import gradient2

"""완전 연결 다층 신경망(확장판)
    가중치 감소, 드롭아웃, 배치 정규화 구현
    Parameters
    ----------
    input_size : 입력 데이터 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    weight_decay_lambda : 가중치 감소(L2 법칙)의 세기
    use_dropout : 드롭아웃 사용 여부
    dropout_ration : 드롭아웃 비율
    use_batchNorm : 배치 정규화 사용 여부
    """
class MultiLayerNetExtend:
    def __init__(self, input_size, hidden_size_list, output_size
                 , activation='relu', weight_init_std='relu'
                 , weight_decay_lambda=0
                 , use_dropout=False
                 , dropout_ration = 0.5
                 , use_batchnorm=False):

        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.output_size = output_size

        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.params = {}

        # 가중치 초기화
        self.__init_weight(weight_init_std)

        # 계층 생성
        activation_layer = {'sigmoid': layers.Sigmoid, 'relu': layers.Relu}
        self.layers = OrderedDict()

        # < 은닉층 생성 >
        # self.hidden_layer_num 개수만큼
        for idx in range(1, self.hidden_layer_num + 1):
            # (1) Affine 계층
            self.layers['Affine' + str(idx)] = layers.Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

            # (2) BatchNormalization 계층
            if self.use_batchnorm:
                # 각 계층별 배치 정규화 계층에서 사용할 매개변수 최기화
                # 원본 그대로에서 시작하는 것으로 초기화. 1배 확대(gamma), 이동 0(beta)
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx - 1])    # 1
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx - 1])    # 0
                self.layers['BatchNorm' + str(idx)] = layers.BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])

            # (3) 활성화 함수
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

            # (4) Dropout 계층
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = layers.Dropout(dropout_ration)

        # < 출력층 Affine 생성 >
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = layers.Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        # < 출력층 생성 >
        # 출력층 활성화 함수로 Softmax, 손실함수로 cross_entropy_error 사용
        self.last_layer = layers.SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """가중치 초기화

        Parameters
        ----------
        weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
            'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
            'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
        """
        # 입력층 노드 수를 리스트로 변형 + self.hidden_size_list는 이미 리스트 형태 이므로 더하기만 + 출력층 노드 수를 리스트로 변형
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        # (모든 계층 수 - 1) 개 만큼
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            # 앞 계층의 노드 수를 고려한 가중치의 표준편차
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            # 가중치의 표준편차 적용한(앞 계층의 노드 수 X 뒷 계층의 노드 수) 형상으로 매개변수 초기화
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            # < 계층에 배치 정규화나 드롭아웃 계층을 추가 했다면 처리해준다. >
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        """손실 함수를 구한다.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x, train_flg)

        """가중치 제곱 법칙(L2)에 따른 가중치 감소
        weight_decay_lambda : 정규화의 세기를 조절하는 하이퍼파라미터.
                              크게 설정할 수록 큰 가증치에 대한 패널티가 커진다.
        """
        weight_decay = 0
        # hidden_layer_num + 1개 만큼 (출력층의 Affine 계층까지 포함해줌)
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            # 1/2 * lambda * (가중치 제곱후 각 원소를 모두 더한 값)
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        # 손실함수_new = 손실함수_old + 가중치 감소
        # last_layer로 SoftmaxWithLoss 사용 -> 오차역전파법에 weight_decay 미분한 값(lambda * W)이 더해진다.
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, X, T):
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)

        if T.ndim != 1:
            T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    def gradient(self, x, t):
        """기울기를 구한다

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        loss = self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        # hidden_layer_num + 1 만큼 (출력층 Affine 포함)
        for idx in range(1, self.hidden_layer_num + 2):
            # 각 Affine 층의 가중치 매개변수에 가중치 감소의 미분값을 더해준다.
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params['W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            # < BatchNormalization 계층 사용한다면 매개변수 갱신해준다.>
            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads

    def numerical_gradient(self, X, T):
        """기울기를 구한다(수치 미분).

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        loss_W = lambda W: self.loss(X, T, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            # 가중치 매개변수에 가중치 감소의 미분값을 더해준다.
            grads['W' + str(idx)] = gradient2.numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = gradient2.numerical_gradient(loss_W, self.params['b' + str(idx)])

            # < BatchNormalization 계층 사용한다면 매개변수 갱신해준다.>
            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = gradient2.numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = gradient2.numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads