import sys, os
sys.path.append('/content/drive/modu/deep_learning/')
from collections import OrderedDict
from modu_labs.modu_layers import *


class LeNet:
    """
    네트워크 구성은 아래와 같음
        conv - relu - pool-
        conv - relu - pool -
        affine - relu -
        affine - softmax
    """

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1={'filter_num': 6, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 conv_param_2={'filter_num': 16, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size_list=[100],
                 output_size=10,
                 weight_init_std=0.01):
        self.conv_layer_num = len([conv_param_1, conv_param_2])
        self.affine_layer_size_list = hidden_size_list + [output_size]
        self.conv_affine_layer_num = self.conv_layer_num + len(self.affine_layer_size_list)

        self.params = {}
        pre_channel_num = input_dim[0]

        # < 가중치 초기화 >
        for idx, conv_param in enumerate([conv_param_1, conv_param_2]):
            self.params['W' + str(idx + 1)] = weight_init_std * np.random.randn(conv_param['filter_num'],
                                                                                pre_channel_num,
                                                                                conv_param['filter_size'],
                                                                                conv_param['filter_size'])
            self.params['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']

        self.params['W3'] = weight_init_std * np.random.randn(16 * 4 * 4, hidden_size_list[0])
        self.params['b3'] = np.zeros(hidden_size_list[0])
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size_list[0], output_size)
        self.params['b4'] = np.zeros(output_size)

        # < 계층 생성 >
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param_1['stride'],
                                           conv_param_1['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], conv_param_2['stride'],
                                           conv_param_2['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return (acc / x.shape[0]) * 100  # 백분률 (%)

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 기울기 계산 결과
        grads = {}
        for idx in range(1, self.conv_layer_num + 1):
            grads['W' + str(idx)] = self.layers['Conv' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Conv' + str(idx)].db

        for idx in range(self.conv_layer_num + 1, self.conv_affine_layer_num + 1):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
