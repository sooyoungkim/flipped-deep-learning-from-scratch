import numpy as np
import matplotlib.pylab as plt

"""출력층 함수 : 소프트맥스 함수

    :parameter
    ----------------
    x : 입력 데이터
    
    :return
    ----------------
    
"""
def softmax(x):
    # 원소별 최대값을 빼준다.
    x -= np.max(x)
    # 원소별 지수함수를 취한다.
    exp_a = np.exp(x)
    # 그 합을 구한다.
    sum_exp_a = np.sum(exp_a)

    return exp_a / sum_exp_a


############################################
# (1) 소프트맥스 함수의 출력은 0 ~ 1 사이의 실수이다.
# (2) 소프트맥스 함수의 출력의 총합은 1이다.
############################################
data = np.array([1010, 1000, 990])

# 배열 a의 최대값인 원소를 찾는다.
m = np.max(data)
print(m)                # 1010

# 지수함수
exp_a = np.exp(data - m)
print(exp_a)            # [1.00000000e+00 4.53999298e-05 2.06115362e-09]

# 지수함수의 합
sum_exp_a = np.sum(exp_a)
print(sum_exp_a)        # 1.0000454019909162

# 합으로 각각 나누기
res = exp_a / sum_exp_a
print(res)              # [9.99954600e-01 4.53978686e-05 2.06106005e-09]

# 소프트맥스 함수 호출
y = softmax(data)
print(y)                # [9.99954600e-01 4.53978686e-05 2.06106005e-09]
print(np.sum(y))        # 1.0


x = np.arange(-5.0, 5.0, 0.1)
y = softmax(x)
plt.plot(x, y)        # x축과 y축 추가
plt.ylim(-0.1, 1.1)     # y축의 범위 지정
plt.show()





