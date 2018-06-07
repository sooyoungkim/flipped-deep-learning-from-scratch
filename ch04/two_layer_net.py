##############################################################################
# 4.5.2 미니 배치 학습
# 학습
# 신경망에는 적응 가능한 가중치와 편향이 있고, 이 가중치와 편향을 훈련 데이터에 적응하도록 조정하는 과정을 '학습'이라 합니다.
##############################################################################
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.functions import *
from common import gradient2 as gradient
# from common import gradient

####################################
#                                  #
# ch04/train_neuralnet 에서 사용해본다.#
#                                  #
####################################
# 신경망에는 적응 가능한 가중치와 편향이 있고, 이 가중치와 편향을 훈련 데이터에 적응하도록 조정하는 과정을 학습이라 합니다.
class TwoLayerNet:
    # 입력층 뉴런수, 은닉층 뉴런수, 출력층 뉴런수
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화 (은닉층 & 출력층)
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  # 앞층 뉴런수 X 다음층 뉴런 수
        self.params['b1'] = np.zeros(hidden_size)                                       # 앞층 뉴런이 없으므로 다음층 뉴런수만 고려
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 추론 : 예측하기
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # 손실함수, 비용함수
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        # print(x.shape)                    # (batch_size, 784)
        # print(t.shape)                    # (batch_size, 10)
        y = self.predict(x)                 # 추론 : 예측하기

        return cross_entropy_error(y, t)    # 손실 값 구하기

    def accuracy(self, x, t):
        # todo  if x.ndim == 1: 처리빠짐 <---- 여기서는 배치처리를 기본으로 하고 짜여진듯.
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    ###################################
    # 엄청 느리다 !!!!!                  #
    ###################################
    # 기울기 구하기 & 업데이트 (방법1. 수치미분)
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        # print(x.shape)  # (batch_size, 784)
        # print(t.shape)  # (batch_size, 10)

        # 입력데이터, 정답레이블은 동일하게 유지되면서 매개변수만 갱신된다.
        loss_W = lambda W: self.loss(x, t)  # W는 더미 용도(사용하지 않음) -> 기울기 계산에서 f(x) 호출하는 형태를 맞춰주기 위함

        # 손실 함수의 값을 줄이기위해 각 가중치, 편향 매개변수의 기울기를 구해서 업데이트한다.
        grads = {}
        grads['W1'] = gradient.numerical_gradient(loss_W, self.params['W1'])    # loss_W(self.params['W1']) -> W1 미분값
        grads['b1'] = gradient.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = gradient.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = gradient.numerical_gradient(loss_W, self.params['b2'])

        # # 갱신하려는 기울기값
        # print("grads['W1'] : ", grads['W1'])
        # # 갱신하기 전에는 기존 기울기 매개변수값 그대로이다.
        # print("self.params['W1'] : ", self.params['W1'])

        return grads

    # 기울기 구하기 & 업데이트 (방법2. 역전파)
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        # todo  if x.ndim == 1: 처리빠짐 <---- 여기서는 배치처리를 기본으로 하고 짜여진듯.
        batch_num = x.shape[0]

        # forward -> predict() 함수와 동일한 역할
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)     # 신경망이 예측한 값

        # backward

        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads