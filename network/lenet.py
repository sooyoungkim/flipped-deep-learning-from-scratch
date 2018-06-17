import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
from collections import OrderedDict
from common.layers import *


class LeNet:
    """
    네트워크 구성은 아래와 같음
        conv - relu - pool-
        conv - relu - pool -
        affine - relu - dropout -
        affine - dropout - softmax
    """

    def __init__(self, input_dim=(1, 28, 28),  # 입력데이터 차원 수
                 conv_param_1={'filter_num': 6, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 conv_param_2={'filter_num': 16, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size_list=[100],
                 output_size=10):
        self.conv_layer_num = len([conv_param_1, conv_param_2])
        self.affine_layer_size_list = hidden_size_list + [output_size]
        self.conv_affine_layer_num = self.conv_layer_num + len(self.affine_layer_size_list)

        # 가중치 초기값
        # 각 층의 뉴런(노드) 하나당 "앞 층의 몇 개 뉴런"과 연결되는가（채널 * 필터크기)
        pre_node_nums = np.array([1 * 5 * 5, 6 * 5 * 5, 16 * 4 * 4, hidden_size_list[0]])
        wight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLU를 사용할 때의 권장 초깃값

        self.params = {}
        pre_channel_num = input_dim[0]  # 입력데이터의 채널 수로 초기화

        # < 가중치 초기화 >
        # conv 계층 필터 초기화
        #   입력 데이터(C, H, W) * 필터(FN, C, FH, FO) = 출력(FN, OH, OW) + 편향(FN, 1, 1) = 출력(FN, OH, OW)
        for idx, conv_param in enumerate([conv_param_1, conv_param_2]):
            self.params['W' + str(idx + 1)] = wight_init_scales[idx] * np.random.randn(conv_param['filter_num'],    # FN
                                                                                       pre_channel_num,             # C = 이전 계층의 출력 채널 수
                                                                                       conv_param['filter_size'],   # FH
                                                                                       conv_param['filter_size'])   # FO
            self.params['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])    # FN
            pre_channel_num = conv_param['filter_num']                              # 필터를 FN개 적용하면 -> 출력 feature 맵도 FN개 생성, 출력 채널 수 = FN -> 다음층의 채널수로 입력된다.


        # affine 계층 가중치 초기화
        self.params['W3'] = wight_init_scales[2] * np.random.randn(16 * 4 * 4, hidden_size_list[0])
        self.params['b3'] = np.zeros(hidden_size_list[0])
        self.params['W4'] = wight_init_scales[3] * np.random.randn(hidden_size_list[0], output_size)
        self.params['b4'] = np.zeros(output_size)


        # 계층 생성
        self.layers = OrderedDict()
        # 합성곱-풀링 계층
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param_1['stride'], conv_param_1['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], conv_param_2['stride'], conv_param_2['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 은닉층
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Dropout3'] = Dropout(0.5)

        # 출력층
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Dropout4'] = Dropout(0.5)
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            # < 계층에 드롭아웃 계층을 추가 했다면 >
            if "Dropout" in key:
                x = layer.forward(x, train_flg)
            else:
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
            # 훈련아닌 검증 모드
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

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

        # 결과 저장
        grads = {}
        for idx in range(1, self.conv_layer_num + 1):
            grads['W' + str(idx)] = self.layers['Conv' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Conv' + str(idx)].db

        for idx in range(self.conv_layer_num + 1, self.conv_affine_layer_num + 1):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

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
        for key, val in params.items():
            self.params[key] = val

        for conv_id in range(1, self.conv_layer_num + 1):
            self.layers['Conv' + str(conv_id)].W = self.params['W' + str(conv_id)]
            self.layers['Conv' + str(conv_id)].b = self.params['b' + str(conv_id)]

        for affine_id in range(self.conv_layer_num + 1, self.conv_affine_layer_num + 1):
            self.layers['Affine' + str(affine_id)].W = self.params['W' + str(affine_id)]
            self.layers['Affine' + str(affine_id)].b = self.params['b' + str(affine_id)]
