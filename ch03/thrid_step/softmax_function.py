import numpy as np


a = np.array([1010, 1000, 990])

# 배열 a의 최대값인 원소를 찾는다.
t = np.max(a)
print(t)                # 1010

# 지수함수
exp_a = np.exp(a - t)
print(exp_a)            # [1.00000000e+00 4.53999298e-05 2.06115362e-09]

# 지수함수의 합
sum_exp_a = np.sum(exp_a)
print(sum_exp_a)        # 1.0000454019909162

# 소프트맥스
res = exp_a / sum_exp_a
print(res)                # [9.99954600e-01 4.53978686e-05 2.06106005e-09]


def softmax(x):
    x -= np.max(x)
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)

    return exp_a / sum_exp_a


############################################
# (1) 소프트맥스 함수의 출력은 0 ~ 1 사이의 실수이다.
# (2) 소프트맥스 함수의 출력의 총합은 1이다.
############################################
y = softmax(a)
print(y)            # [9.99954600e-01 4.53978686e-05 2.06106005e-09]
print(np.sum(y))    # 1.0


