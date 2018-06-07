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


"""출력층 활성화 함수 : 소프트맥스 함수

    parameter
    --------------------------------
    x : (Wx + b)의 값, affine 계층 출력값

    :return
    --------------------------------
    분류 class 개수 만큼 (= 출력층 size)
"""
def softmax(x):
    if x.ndim == 2:     # 2 차원이면
        x = x.T         # 입력 데이터를 transpose 한다 -> x를 transpose 하지 않으면 아래 뺄셈시에 형상이 달라서 error 발생
        x = x - np.max(x, axis=0)   # 오버플로 대책 : 행렬 x 의 0차원(column)마다 max 값을 구해서 각 원소마다 빼준다.
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T      # 다시 원래 형상으로 복구

    # 지수함수값이 쉽게 아주 큰 값이 되는 문제를 해결하는 방법 : 입력 신호 중 최대값을 각 원소마다 빼주는 것이다.
    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


"""손실 함수 (cost,loss function) : 평균 제곱 오차(MSE)

    : parameters
    --------------------------------
    y : 신경망이 예측한 출력값 (넘파이 배열), 
    t : 정답 레이블 (넘파이 배열)

    : return 
    ---------------------------------
    손실 값 (scalar)
"""
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)


"""손실 함수 (cost,loss function) : 교차 엔트로피 오차(CEE)
    : 정답 일때의 추정값(예측값)에 대해 자연로그를 계산한다.


    : parameters
    --------------------------------
    y : 신경망이 예측한 출력값 (넘파이 배열), 
    t : 정답 레이블 (넘파이 배열)
    
    :return
    ---------------------------------
    손실 값 (scalar)
    
"""
def cross_entropy_error(y, t):      # (y:신경망의 출력값(예측값) 넘파이 배열, t:정답 레이블 넘파이 배열)
    # 1차원 배열이면, 즉 처리할 데이터가 하나인 경우(이미지 1장에 대한 넘파이 배열인 경우)
    if y.ndim == 1:
        t = t.reshape(1, t.size)    # 형상 변형시키기 - 예) 1차원 배열 [1 2 3 4] 형상 (4,)  -> 2차원 배열 [[1 2 3 4]] 형상 (1,4) 로 변형
        y = y.reshape(1, y.size)

    # 정답 레이블 t가 원-핫 인코딩 벡터라면(즉, 정답을 가리키는 위치의 원소는 1로 표시됨 (그 외에는 0으로 표기))
    if t.size == y.size:
        t = t.argmax(axis=1)    # t 행렬의 각 1차원(row)을 축으로 max인 원소(여기서는 1)들의 인덱스를 리스트로 반환

    # 데이터가 여러개인 경우(배치처리) & 정답 레이블이 정답 인덱스만 담긴 1차원 넘파이 배열 형태인 경우
    # 입력값의 데이터 수를 배치 사이즈로 사용
    batch_size = y.shape[0]

    # t는 원-핫 인코딩이 아니라 정답 인덱스만 담긴 1차원 넘파이 배열 형태이다.
    # np.log() 함수에 0을 입력하면 마이너스 무한대가(-inf)가 되므로 0이 되지 않도록 아주 작은 delta 값(1e-7)을 더해준다.
    # 배치 크기로 나누어 정규화 & 이미지 1장당 평균의 오차를 계산한다.
    # 정답에 해당하는 신경망의 출력값만 가지고 교차 엔트로피 오차를 계산한다.
    # y[np.arange(batch_size), t] -> zip 형태로 인덱스들이 묶이고, 해당 인덱스의 값을 모두 더한다.
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# softmax + 손실함수
def softmax_loss(x, t):
    y = softmax(x)
    return cross_entropy_error(y, t)


if __name__ == '__main__':

    """softmax 테스트"""
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print(x)
    # [[1 2 3]
    #  [4 5 6]]
    x = x.T  # 입력 데이터를 transpose 하고
    print(x)
    # [[1 4]
    #  [2 5]
    #  [3 6]]
    print(np.max(x, axis=0))    # max 값을 구하면 (2,)가 된다.
    # [3 6]
    x = x - np.max(x, axis=0)   # x를 transpose 하지 않으면 형상이 달라서 뺄셈시에 error 발생 : (2,3) (2,)
    print(x)
    # [[-2 - 2]
    #  [-1 - 1]
    #  [0  0]]
    print(np.sum(np.exp(x), axis=0))
    # [1.50321472 1.50321472]
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    print(y)
    # [[0.09003057 0.09003057]
    #  [0.24472847 0.24472847]
    #  [0.66524096 0.66524096]]
    print(y.T)
    # [[0.09003057 0.24472847 0.66524096]
    #  [0.09003057 0.24472847 0.66524096]]

    """ValueError: operands could not be broadcast together with shapes (2,3) (2,)"""
    # x = np.array([[1, 2, 3], [4, 5, 6]])
    # print(np.max(x, axis=1))  행렬 x 의 1차원(row)마다 max 값을 구해서 각 원소마다 빼준다.
    # x = x - np.max(x, axis=1)