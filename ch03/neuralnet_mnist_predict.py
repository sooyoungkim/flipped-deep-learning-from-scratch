##############################################################################
# 간단한 신경망 "추론 처리"
#
# 100 ~ 101 페이지 : MNIST 데이터를 가져와서 추론(정답 레이블과 비교하여 정확도 구하기)
#   - 훈련 이미지가 60,000장, 시험 이미지가 10,000장
#       => 훈련 이미지를 사용하여 모델을 학습하고,
#           학습한 모델로 시험 이미지들을 얼마나 정확하게 분류하는지를 평가한다.
#
##############################################################################
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset import mnist
from common import functions


def get_data():
    # 데이터 가져오기 : (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    #   - normalize=True 로 설정하면 0 ~ 255 범위인 각 픽셀의 값을 0.0 ~ 1 범위로 변환하여 반환해준다.
    #       - 신경망의 입력 데이터에 특정 변환을 가하는 것을 전처리라 한다.
    #       - 입력 이미지 데이터에 대한 전처리 작업으로 정규화를 수행한셈이다.
    (x_train, t_train), (x_test, t_test) = mnist.load_mnist(normalize=True, flatten=True, one_hot_label=False)

    # 테스트용(검증용) 이미지와 레이블만 반환한다. -> 여기서는 신경망 학습 구현이 아닌 추론 처리만 구현하므로 시험 데이터만 필요하다.
    return x_test, t_test


def init_network():
    # 이미 학습된 매개변수(가중치, 편향)가 저장된 피클파일 읽어오기
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    # W1.shape : (784, 50), W1.shape : (50, 100), W1.shape : (100, 10)
    #   => 입력층 뉴런수 : 784 -> 은닉층1의 뉴런수 : 50 -> 은닉층2의 뉴런수 : 100 -> 출력층 뉴런수 : 10 개로 구성됨을 알 수 있다.
    print("W1.shape : {}, W1.shape : {}, W1.shape : {}".format(network['W1'].shape
                                                               , network['W2'].shape
                                                               , network['W3'].shape))
    return network


"""신경망이 예측한 결과값

    parameter
    ---------------------------------
    network : 가중치 매개변수 담긴 딕셔너리
    x : 입력 데이터 

    :return
    ---------------------------------
    y : 신경망이 예측한 결과값
"""
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = functions.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = functions.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    # 신경망을 이용한 분류에서는 일반적으로 가장 큰 출력을 내는 뉴런에 해당하는 클래스로만 인식한다.
    # 소프트맥스 함수를 적용해도 가장 큰 뉴런의 위치는 달라지지 않는다.(지수함수를 사용하므로)
    # 추론 단계에서는 출력층의 소프트맥스 함수를 생략하는 것이 일반적이다.(신경망 학습시에는 사용한다.)
    #   - 지수함수를 사용해서 학습시에 결과 값의 차이를 극명하게 나타나게하는 역할을 한다.
    # (생략)
    # y = functions.softmax(a3)
    y = a3

    return y


x, t = get_data()
print(x.shape)  # (10000, 784)
print(t.shape)  # (10000,)

network = init_network()
accuracy_cnt = 0


# 10,000 장의 이미지를 한장씩 가져와서 10,000회 반복하여 처리
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)        # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:           # 신경망이 예측한 답변과 정답 레이블을 비교
        accuracy_cnt += 1   # 맞힌 수 세기

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
# (맞힌 예측 수 / 전체 이미지 갯수) -> 정확도 구한다. Accuracy:0.9352
