##############################################################################
#
# 104 페이지 : MNIST 데이터를 가져와서 배치로 처리하여 추론하기
#   - 훈련 이미지가 60,000장, 시험 이미지가 10,000장
#       => 훈련 이미지를 사용하여 모델을 학습하고,
#           학습한 모델로 시험 이미지들을 얼마나 정확하게 분류하는지를 평가한다.
#
##############################################################################
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    # 데이터 가져오기 : (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    _, (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


"""신경망이 예측한 결과값
    
    parameter
    ---------------------------------
    network : 가중치 매개변수 담긴 딕셔너리
    x : 입력 데이터 
    
    :return
    ---------------------------------
    y : 신경망이 예측한 결과값 (10개의 클래스로 분류된 데이터, 0 ~ 9)
"""
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    # 추론 단계에서는 출력층의 소프트맥스 함수를 생략하는 것이 일반적이다.(신경망 학습시에는 사용한다.)
    # (생략)
    # y = softmax(a3)
    y = a3

    return y


x, t = get_data()

network = init_network()
batch_size = 100    # 배치 크기
accuracy_cnt = 0

# 10,000 장의 이미지를 100개씩 묶어 가져와서 100회 반복하여 처리 (에폭 : 100)
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]             # (100, 784) -> 입력 데이터 i번 ~ (i + bacth_size - 1)번 까지 포함한 배치 데이터
    y_batch = predict(network, x_batch)     # (100, 10)  -> 신경망이 예측한 결과값
    p = np.argmax(y_batch, axis=1)          # 각 row에서 확률이 가장 높은 원소의 인덱스를 얻는다.
    # (1) 신경망이 예측한 답변과 정답 레이블을 비교 : p == t[i:i+batch_size] 결과는 bool 배열(bacth_size,)이다.
    # (2) 맞힌 수 세기 : 이 bool 배열을 사용해 True인 원소의 수를 합하면 맞힌 수이다.
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
# (맞힌 예측 수 / 전체 이미지 갯수) -> 정확도 구한다. Accuracy:0.9352


#############################################################################
# argmax()의 axis=1 인수의 의미를 알아보자.
#   : axis=1 는 행렬의 1번째 차원(row)을 축으로 최대값의 인덱스를 찾는다는 의미이다.
#     (axis=0인 경우는 열 연산, axis=1인 경우는 행 연산이다. 디폴트 값은 axis=0 이다)
#############################################################################
Z = np.array([[0.1, 0.8, 0.1],      # 1행에서 가장 큰 값(0.8)의 인덱스는 1
              [0.3, 0.1, 0.6],      # 2행에서 가장 큰 값(0.6)의 인덱스는 2
              [0.2, 0.5, 0.3],      # 3행에서 가장 큰 값(0.5)의 인덱스는 1
              [0.8, 0.1, 0.1]])     # 4행에서 가장 큰 값(0.8)의 인덱스는 0
zp = np.argmax(Z, axis=1)
print(zp)             # [1 2 1 0]

#############################################################################
# (1) True 인 원소의 개수를 모두 더한다.
print(np.sum(np.array([ True, False, True, False, False])))  # 2
print(np.sum(np.array([ True, False, True, True, False])))   # 3

ZZ = np.array([[51, 55],
               [14, 19],
               [0, 4]])
print(np.array(ZZ > 15))
# [[ True  True]
#  [False  True]
#  [False False]]
print(np.sum(np.array(ZZ > 15)))  # 3

# (2) [비교] 특정 조건을 만족하는 원소만 얻는다.
print(ZZ[ZZ > 15])                # [51 55 19]
#############################################################################

