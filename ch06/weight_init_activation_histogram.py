##############################################################################
# 6.2.2 은닉층의 활성화값 분포
# 활성화 함수로 시그모이드 함수를 사용하는 5층 신경망에 무작위로 생성한 입력 데이터를 흘리며
#       각 층의 활성화값 분포를 히스토그램으로 그려봅니다.
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
from common import functions

x = np.random.randn(1000, 100)  # 1000개의 데이터 (1000 X 100)
node_num = 100          # 각 은닉층의 노드(뉴런 수)
hidden_layer_size = 5   # 은닉층 5개
activations = {}        # 활성화 결과를 저장

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]    # 이전 계층의 활성화 결과를 입력으로 사용 (1000, 100)

    # < 무작위로 설정된 초기 값을 다양하게 바꿔가며 실험해보자 >
    # (1) 가중치의 표준편차 1
    W = np.random.randn(node_num, node_num) * 1

    # (2) 가중치의 표준편차 0.01
    # W = np.random.randn(node_num, node_num) * 0.01

    # (3) Xavier 초기값 사용 (활성와 함수로 sigmoid 사용시)
    # W = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)

    # (4) He 초기값 사용 (활성화 함수로 ReLU 사용시)
    # W = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    # x(1000, 100) dot W(100, 100) = A(1000, 100)
    A = np.dot(x, W)

    # < 활성화 함수도 다양하게 바꿔가며 실험해보자 >
    Z = functions.sigmoid(A)
    # Z = functions.relu(A)
    # Z = functions.tanh(A)

    # hidden_layer_size 만큼 활성화값이 저장된다. (1000, 100)
    activations[i] = Z


# 히스토그램 그리기
# len(activations) = 5
# [[(1000 X 100)],
#  [(1000 X 100)],
#  [(1000 X 100)],
#  [(1000 X 100)],
#  [(1000 X 100)]]
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    plt.hist(a.flatten(), 30, range=(0, 1))  # a를 1차원으로 flat 하게 편다. (100,000 개 데이터)
plt.show()