##############################################################################
# 4.5.2 미니 배치 학습
# 4.5.3
#
# 학습
# 신경망에는 적응 가능한 가중치와 편향이 있고, 이 가중치와 편향을 훈련 데이터에 적응하도록 조정하는 과정을 '학습'이라 합니다.
#
# 경사하강법을 사용하여 신경망 학습이 이뤄지는 순서
# 1단계 - 미니배치
#           훈련 데이터 중 일부를 무작위로 가져옵니다. 이렇게 선별한 데이터를 미니배치라하며,
#           "그 미니배치의 손실 함수 값을 줄이는 것"이 목표입니다.
# 2단계 - 기울기 산출
#           미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구합니다.
#           기울기는 손실 함수의 값을 가장 작게하는 방향을 제시합니다.
# 3단계 - 매개변수 갱신
#           가중치 매개변수를 기울기 방향으로 일정 거리만큼 갱신합니다.
# 4단계 - 반복
#           1 ~ 3 단계를 반복합니다.
##############################################################################
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from ch04 import two_layer_net

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = two_layer_net.TwoLayerNet(input_size=784, hidden_size=2, output_size=10)   #todo test
# network = two_layer_net.TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 3               # todo test
batch_size = 10             # todo test
# iters_num = 10000               # 반복 횟수
# batch_size = 100                # 미니배치 크기
train_size = x_train.shape[0]   # 60000
learning_rate = 0.1             # 학습률

# 학습 경과 히스토리
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(1, 1)  # todo test
# iter_per_epoch = max(train_size / batch_size, 1)

# 경사하강법 1~4
# (4/4) 훈련 데이터 중 일부를 무작위로 추출하여 매개변수 갱신하는 작업을 반복한다.
for i in range(iters_num):
    # (1/4) 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)   # 0 이상 60,000 미만의 수 중에서 무작위로 100개 추출
    x_batch = x_train[batch_mask]   # 훈련용 이미지 데이터
    t_batch = t_train[batch_mask]   # 훈련용 정답 레이블

    # (2/4) 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)     # 수치미분
    # grad = network.gradient(x_batch, t_batch)             #

    # (3/4) 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()