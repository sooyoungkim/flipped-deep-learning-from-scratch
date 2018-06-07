##############################################################################
# 7.5 CNN 구현하기
# “Convolution-ReLU-Pooling-Affine-ReLU-Affine-Softmax” 순으로 흐르는
# 단순한 합성곱 신경망(CNN)입니다.
##############################################################################
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient2 import numerical_gradient


class SimpleConvNet:
    """단순한 합성곱 신경망

    conv - relu - pool - affine - relu - affine - softmax

    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    """

    # 채널 1개, 28 X 28 데이터 처리
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10,
                 weight_init_std=0.01):
        filter_num = conv_param['filter_num']  # conv 계층에서 FN 만큼 출력된다
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        # 입력 데이터의 채널 수 그대로 출력 데이터로 내보낸다. (채널마다 독립적으로 계산)
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # (1) 가중치 초기화
        self.params = {}
        # conv계층 필터 파라미터
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)

        # < pool 계층은 학습할 매개변수가 없다 >

        # affine 계층 가중치 파라미터
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        # 마지막 affine 계층 가중치 파라미터
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # (2) CNN 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        # 계층 맨 앞에서부터 차례로 forward()호출 (결과를 다음 계층의 입력으로 전달)
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """손실 함수를 구한다.
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x)
        # 예측 값(predict)을 손실함수의 입력값으로 사용
        #   즉, 계층 맨 앞에서부터 마지막 계층까지 forward를 처리한다.
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        """기울기를 구한다（수치미분）.
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
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).
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
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
            # print(params)
        for key, val in params.items():
            # print("key : ", key, ", value : ", val)
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]


if __name__ == '__main__':
    # Conv 계층 테스트
    # 데이터 1, 채널 2, 높이 7, 너비 7
    x = np.array([[[[111, 112, 113, 114, 115, 116, 117],
                    [121, 122, 123, 124, 125, 126, 127],
                    [131, 132, 133, 134, 135, 136, 137],
                    [141, 142, 143, 144, 145, 146, 147],
                    [151, 152, 153, 154, 155, 156, 157],
                    [161, 162, 163, 164, 165, 166, 167],
                    [171, 172, 173, 174, 175, 176, 177]],
                   [[211, 212, 213, 214, 215, 216, 217],
                    [221, 222, 223, 224, 225, 226, 227],
                    [231, 232, 233, 234, 235, 236, 237],
                    [241, 242, 243, 244, 245, 246, 247],
                    [251, 252, 253, 254, 255, 256, 257],
                    [261, 262, 263, 264, 265, 266, 267],
                    [271, 272, 273, 274, 275, 276, 277]]
                   ]])
    input_dim = (2, 7, 7)

    weight_init_std = 0.01
    filter_num = 1
    filter_size = 3
    filter_stride = 2
    filter_pad = 0
    params = {}
    params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
    params['b1'] = np.zeros(filter_num)

    print("x.size : ", x.size)
    # x.size: 98

    conv = Convolution(params['W1'], params['b1'], filter_stride, filter_pad)
    print(conv.forward(x))
    # [[[[10.74446424 10.86632772 10.9881912]
    #    [11.96309901 12.08496249 12.20682597]
    #   [13.18173378 13.30359726 13.42546074]]]]

    relu = Relu()
    print(relu.forward(conv.forward(x)))
    # [[[[10.74446424 10.86632772 10.9881912]
    #    [11.96309901 12.08496249 12.20682597]
    #   [13.18173378 13.30359726 13.42546074]]]]

    pool = Pooling(pool_h=2, pool_w=2, stride=1)
    print(pool.forward(relu.forward(conv.forward(x))))
    # [[[[12.08496249 12.20682597]
    #    [13.30359726 13.42546074]]]]


