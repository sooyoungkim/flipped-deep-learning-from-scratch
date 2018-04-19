##############################################################################
# 4.2 손실함수
# 4.2.1 평균 제곱 오차(MSE)
# 4.2.2 교차 엔트로피 오차(CEE)
##############################################################################
import numpy as np
from common.functions import mean_squared_error, cross_entropy_error

# 정답은 '2' <= 2번 index 값을 1로 세팅 (one-hot encoding)
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]


# 예1 : '2'일 확률이 가장 높다고 추정 <= 2번 index 값이 0.6으로 가장 높다.
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t)))
# 0.09750000000000003
print(cross_entropy_error(np.array(y), np.array(t)))
# 0.510825457099338


# 예2 : '7'일 확률이 가장 높다고 추정 <= 7번 index 값이 0.6으로 가장 높다.
#
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t)))
# 0.5975
print(cross_entropy_error(np.array(y), np.array(t)))    # cross_entropy_error 사용하는 경우 정답이 t의 2번째 index이므로 y의 2번째만 계산한 것과 같다.
# 2.302584092994546


# => 결과(오차값)가 더 작은 첫번째(예1) 추정이 정답일 가능성이 높다고 판단한다.

