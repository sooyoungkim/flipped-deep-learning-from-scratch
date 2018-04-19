# coding: utf-8
import numpy as np


# 활성화 함수 : 계단함수
def step_function(x):
    return np.array(x > 0, dtype=np.int)


# 활성화 함수 : 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


# 활성화 함수 : 렐루 함수
def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


# 출력층 활성화 함수 : 항등 함수
def identity_function(x):
    return x


# 출력층 활성화 함수 : 소프트맥스 함수
def softmax(x):
    if x.ndim == 2:     # 2 차원이면
        x = x.T
        x = x - np.max(x, axis=0)   # x행렬의 각 0차원(column)에 대해 max 원소의 인덱스
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


# 손실 함수 (cost,loss function) : 평균 제곱 오차(MSE)
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)


# 손실 함수 (cost,loss function) : 교차 엔트로피 오차(CEE)
def cross_entropy_error(y, t):      # (y:신경망의 출력값(예측값) 넘파이 배열, t:정답 레이블 넘파이 배열)
    # 1차원 배열이면, 즉 처리할 데이터가 하나인 경우(이미지 1장에 대한 넘파이 배열인 경우)
    if y.ndim == 1:
        t = t.reshape(1, t.size)    # 형상 변형시키기 - 예) 1차원 배열 [1 2 3 4] 형상 (4,)  -> 2차원 배열 [[1 2 3 4]] 형상 (1,4) 로 변형
        y = y.reshape(1, y.size)

    # 정답 레이블 t가 원-핫 인코딩 벡터라면 -> 정답을 가리키는 위치의 원소는 1로 표시됨 (그 외에는 0으로 표기)
    if t.size == y.size:
        t = t.argmax(axis=1)    # t행렬의 각 1차원(row)을 축으로 max인 원소(여기서는 1)의 인덱스를 리스트로 반환

    batch_size = y.shape[0]

    # t는 원-핫 인코딩이 아니라 정답 인덱스만 담긴 1차원 넘파이 배열 형태이다.
    # np.log() 함수에 0을 입력하면 마이너스 무한대가(-inf)가 되므로 0이 되지 않도록 아주 작은 delta 값(1e-7)을 더해준다.
    # 배치 크기로 나누어 정규화 & 이미지 1장당 평균의 오차를 계산한다.
    # 정답에 해당하는 신경망의 출력값만 가지고 교차 엔트로피 오차를 계산한다.
    # y[np.arange(batch_size), t] -> zip 형태로 반환
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


#
def softmax_loss(x, t):
    y = softmax(x)
    return cross_entropy_error(y, t)
