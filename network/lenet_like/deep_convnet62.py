import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.layers import *
import pickle
from collections import OrderedDict


class DeepConvNet:
    """
    네트워크 구성은 아래와 같음 no.6-2
        conv - relu - pool - dropout -
        conv - relu - pool - dropout -
        conv - relu - pool - dropout -
        affine625 - relu - dropout -
        affine10 - softmax

        모두연
    """

    def __init__(self, input_dim=(1, 28, 28),  # 입력데이터 차원 수
                 # 32 * ((28 + 2*1 - 3)/1 + 1) * = 32 * 28 * 28 -> pooling 적용하면 -> 32 * ((28 - 2)/2 + 1) *  = 32 * 14 * 14
                 # 64 * ((14 + 2*2 - 3)/1 + 1) * = 64 * 16 * 16 -> pooling 적용하면 -> 64 * ((16 - 2)/2 + 1) *  = 64 * 8 * 8
                 # 128 * ((8 + 2*1 - 3)/1 + 1) * = 128 * 8 * 8 -> pooling 적용하면 -> 128 * ((8 - 2)/2 + 1) *  = 128 * 4 * 4
                 conv_param_1={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_2={'filter_num': 64, 'filter_size': 3, 'pad': 2, 'stride': 1},
                 conv_param_3={'filter_num': 128, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 hidden_size=625,
                 output_size=10):
        conv_params = [conv_param_1, conv_param_2, conv_param_3]
        pool_param = {'filter_size': 2, 'pad': 0, 'stride': 2}
        self.pool_h, self.pool_w, self.pool_pad, self.pool_stride = pool_param['filter_size'], pool_param['filter_size'], pool_param['pad'], pool_param['stride']
        self.conv_layer_num = len(conv_params)
        self.hyper_parameters_num = self.conv_layer_num + 2     # conv + FC 계층의 수

        # 각 층의 뉴런(노드) 하나당 "앞 층의 몇 개 뉴런"과 연결되는지 기록
        pre_node_nums = []
        # conv - pooling 계층을 통과한 데이터의 출력 형상 기록
        FN, H, W = input_dim[0], input_dim[1], input_dim[2]
        for ii in range(0, len(conv_params)):
            FN = conv_params[ii]['filter_num']
            FH, FW = conv_params[ii]['filter_size'], conv_params[ii]['filter_size']
            pad, stride = conv_params[ii]['pad'], conv_params[ii]['stride']

            # [!] 입력 레이어에서 필터가 적용되는 영역 만큼만 다음 레이어의 하나의 뉴런에 연결된다 (입력데이터 채널 X 필터 H X 필터 W)
            prev_channel_num = conv_params[ii - 1]['filter_num'] if ii != 0 else input_dim[0]
            pre_node_nums.append(prev_channel_num * FH * FW)

            # [!!] 각 층의 출력 데이터 형상 계산
            # conv 계층
            H = 1 + int((H + 2 * pad - FH) / stride)
            W = 1 + int((W + 2 * pad - FW) / stride)
            # pooling 계층
            H = 1 + int((H + 2 * self.pool_pad - self.pool_h) / self.pool_stride)
            W = 1 + int((W + 2 * self.pool_pad - self.pool_w) / self.pool_stride)

        # conv - pooling 계층을 모두 통과한 데이터의 출력 사이즈
        self.conv_output_size = FN * H * W
        print(FN, "*", H, "*", W, "=", self.conv_output_size)
        pre_node_nums.append(self.conv_output_size)
        pre_node_nums.append(hidden_size)

        # 가중치 초기값 (He)
        wight_init_scales = np.sqrt(2.0 / np.array(pre_node_nums))
        # todo
        wight_init_scales[0] = 0.01
        print(wight_init_scales)
        # [0.47140452 0.11785113 0.11785113 0.08333333 0.08333333 0.05892557 0.04419417 0.2]

        # < 가중치 초기화 >
        self.params = {}
        pre_channel_num = input_dim[0]  # 입력데이터의 채널 수로 초기화

        # conv 계층 필터 초기화 : 입력 데이터(C, H, W) * 필터(FN, C, FH, FO) = 출력(FN, OH, OW) + 편향(FN, 1, 1) = 출력(FN, OH, OW)
        for idx, conv_param in enumerate(conv_params):
            self.params['W' + str(idx + 1)] = wight_init_scales[idx] * np.random.randn(conv_param['filter_num'],    # FN
                                                                                       pre_channel_num,             # C = 이전 계층의 출력 채널 수
                                                                                       conv_param['filter_size'],   # FH
                                                                                       conv_param['filter_size'])   # FO
            self.params['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])    # FN
            pre_channel_num = conv_param['filter_num']                              # 필터를 FN개 적용하면 -> 출력 feature 맵도 FN개 생성, 출력 채널 수 = FN -> 다음층의 채널수로 입력된다.

        # affine 계층 가중치 초기화
        self.params['W4'] = wight_init_scales[3] * np.random.randn(self.conv_output_size, hidden_size)
        self.params['b4'] = np.zeros(hidden_size)
        self.params['W5'] = wight_init_scales[4] * np.random.randn(hidden_size, output_size)
        self.params['b5'] = np.zeros(output_size)

        # < 계층 생성 >
        self.layers = OrderedDict()
        # Conv - Pooling 계층
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param_1['stride'], conv_param_1['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=self.pool_h, pool_w=self.pool_w, stride=self.pool_stride)
        self.layers['Dropout1'] = Dropout(dropout_ratio=0.3)

        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], conv_param_2['stride'], conv_param_2['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=self.pool_h, pool_w=self.pool_w, stride=self.pool_stride)
        self.layers['Dropout2'] = Dropout(dropout_ratio=0.3)

        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'], conv_param_3['stride'], conv_param_3['pad'])
        self.layers['Relu3'] = Relu()
        self.layers['Pool3'] = Pooling(pool_h=self.pool_h, pool_w=self.pool_w, stride=self.pool_stride)
        self.layers['Dropout3'] = Dropout(dropout_ratio=0.3)

        # 은닉층
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Relu4'] = Relu()
        self.layers['Dropout4'] = Dropout(dropout_ratio=0.3)

        # 출력층
        self.layers['Affine5'] = Affine(self.params['W5'], self.params['b5'])
        self.last_layer = SoftmaxWithLoss()


    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            # < 계층에 드롭아웃 계층을 추가 했다면 >
            if "Dropout" in key or "BatchNorm" in key:
            # if "Dropout" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x


    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)


    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]     # 입력데이터 batch_size 만큼 가져오기 -> i * batch_size 부터 (i + 1) * batch_size -1 까지
            tt = t[i * batch_size:(i + 1) * batch_size]     # 정답레이블 batch_size 만큼 가져오기
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        # return acc / x.shape[0]
        return (acc / x.shape[0]) * 100     # 백분률 (%)


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
        # Conv 계층과 Affine 계층의 매개변수 업데이트 (나머지는 매개변수가 없다)
        grads = {}
        for idx in range(1, self.conv_layer_num + 1):
            grads['W' + str(idx)] = self.layers['Conv' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Conv' + str(idx)].db

        for idx in range(self.conv_layer_num + 1, self.hyper_parameters_num + 1):
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

        for affine_id in range(self.conv_layer_num + 1, self.hyper_parameters_num + 1):
            self.layers['Affine' + str(affine_id)].W = self.params['W' + str(affine_id)]
            self.layers['Affine' + str(affine_id)].b = self.params['b' + str(affine_id)]
