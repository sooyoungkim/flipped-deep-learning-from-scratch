##############################################################################
# 6.1.7 어느 갱신 방법을 이용할 것인가?
# SGD, 모멘텀, AdaGrad, Adam의 학습 패턴을 비교합니다.
##############################################################################

import sys,os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *


# 함수 f
def f(x, y):
    return x ** 2 / 20.0 + y ** 2


# f의 미분 함수
def df(x, y):
    return x / 10.0, 2.0 * y

init_pos = (-7.0, 2.0)


# 초기화
params = {}
params['x'], params['y'] = 0, 0
grads = {}
grads['x'], grads['y'] = 0, 0


"""Optimezer 별로 학습 패턴 비교"""
optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["RMSprop"] = RMSprop(lr=0.2)
# optimizers["Adam"] = Adam(lr=0.3)

# 크게 10번만 가보자
# optimizers["SGD"] = SGD(lr=1.05)
# optimizers["Momentum"] = Momentum(lr=0.8)

# 작게 99번만 가보자
# optimizers["SGD"] = SGD(lr=0.35)
# optimizers["Momentum"] = Momentum(lr=0.033)

# 더 작게...
# optimizers["Momentum 0.01"] = Momentum(lr=0.01)   # 380
# optimizers["Momentum 0.001"] = Momentum(lr=0.001) # 3500




idx = 1
for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]

    # 30회 반복
    for i in range(3500):
        x_history.append(params['x'])
        y_history.append(params['y'])

        # 기울기 최적화 업데이트
        grads['x'], grads['y'] = df(params['x'], params['y'])   # 미분
        optimizer.update(params, grads)     # 기울기 최적화

    # x축과 y축의 값
    x = np.arange(-10, 10, 0.01)  # -10 <= x < 10 (0.01 만큼씩) : x: [-10. - 9.99 - 9.98...   9.97   9.98   9.99]
    y = np.arange(-5, 5, 0.01)    #  -5 <= y <  5 (0.01 만큼씩) : y: [ -5. - 4.99 - 4.98...   4.97   4.98   4.99]

    # x축을 나타내는 점들, y축을 나타내는 점들 => 하나의 그리드
    X, Y = np.meshgrid(x, y)
    # 최적화된 기울기를 함수 f 결과 값
    Z = f(X, Y)

    # 외곽선 단순화
    mask = Z > 7
    Z[mask] = 0  # 결과값 중 7보다 큰 것을(True) 0으로 masking.

    # 그래프 그리기 (2 X 2, )
    plt.subplot(1, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.contour(X, Y, Z)    # X, Y, Z 윤곽 표시 (등고선)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')  # (0,0) 위치 표시
    # colorbar()
    # spring()
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")

plt.show()