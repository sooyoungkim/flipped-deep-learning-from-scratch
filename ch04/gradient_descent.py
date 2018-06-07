##############################################################################
# 4.4.1 경사하강법
# 각 지점에서 함수의 값을 낮추는 방안을 제시하는 지표가 기울기이다.
##############################################################################
import numpy as np
import matplotlib.pylab as plt
from common import gradient2 as gradient


"""경사 하강법
    
    parameter
    -------------------
    f       : 최적화 하려는 함수
    init_x  : 초기값 
    lr      : 학습률
    step_num: 반복횟수
    
    return
    -------------------
    (최적화한 매개변수 값, 매개변수 갱신이력 배열)
    
"""
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x      # 초기값
    x_history = []  # 갱신 과정을 저장

    # step_num 횟수만큼 반복해서 실행
    for i in range(step_num):
        x_history.append(x.copy())

        # 기울기 구하기: 최적화 하려는 함수 f 와 변수 x (각각의 변수에 대해 편미분)
        grad = gradient.numerical_gradient(f, x)
        # (이전 기울기값 - 학습률 * 각 변수의 편미분 값) 으로 갱신
        x -= lr * grad

    return x, np.array(x_history)


# 2개의 변수(x0, x1)로 초기화
init_x = np.array([-3.0, 4.0])
lr = 0.1
step_num = 20


# function_2 최적화를 위해 경사하강법을 사용
x, x_history = gradient_descent(gradient.function_2, init_x, lr=lr, step_num=step_num)
print(x)    # 최적화한 매개변수 값은 [-0.03458765  0.04611686]


# [그림 4-10]
plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()