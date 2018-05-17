##############################################################################
# 4.4.1 경사법(경사하강법)
# 각 지점에서 함수의 값을 낮추는 방안을 제시하는 지표가 기울기이다.
##############################################################################
import numpy as np
import matplotlib.pylab as plt
from common import gradient2 as gradient


# 경사 하강법
#   f : 최적화 하려는 함수
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x      # 최소값
    x_history = []  # 갱신 과정을 저장

    # step_num 횟수만큼 반복해서 실행
    for i in range(step_num):
        x_history.append( x.copy() )

        # 기울기 구하기
        grad = gradient.numerical_gradient(f, x)
        # (이전에 갱신된 값 - 학습률 * 편미분) 한 값으로 갱신
        x -= lr * grad

    return x, np.array(x_history)


# 변수가 두 개인(여기서는 x[0], x[1]) 함수 <- init_x 를 일차원에 원소가 2개로 테스트
def function_2(x):
    return np.sum(x ** 2)   # x[0] ** 2 + x[1] ** 2


# 초기 값
init_x = np.array([-3.0, 4.0])

lr = 0.1
step_num = 20


# 최소값 탐색 시작
# function_2 최적화를 위해 경사하강법을 사용
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
print(x)    # 최소값은 [-0.03458765  0.04611686]


# [그림 4-10]
plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()