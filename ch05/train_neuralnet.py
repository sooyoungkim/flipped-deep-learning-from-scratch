import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset import mnist
from ch05 import two_layer_net

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = mnist.load_mnist(normalize=True, one_hot_label=True)

###############################################
#                                             #
# ch05/two_layer_net 의 TwoLayerNet을 사용해본다. #
#   -   #
#                                             #
###############################################
network = two_layer_net.TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

train_size = x_train.shape[0]   # 훈련 데이터수
batch_size = 100                # 배치크기
iters_num = 10000               # 배치처리 반복 회수
learning_rate = 0.1             # 학습률 (어느 정도의 폭으로 훈련시킬지)
iter_per_epoch = max(train_size / batch_size, 1)    # 1에폭

train_loss_list = []    # 훈련용 손실 데이터
train_acc_list = []     # 훈련용 정확도 history
test_acc_list = []      # 시험용 정확도 history


# 경사하강법
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)  # 배치크기만큼 랜덤 선택
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 오차역전파법으로 기울기를 구한다.
    grad = network.gradient(x_batch, t_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록(Loss 히스토리)
    train_loss = network.loss(x_batch, t_batch)
    train_loss_list.append(train_loss)

    # 1 에폭당 정확도 히스토리 남기기
    if i % iter_per_epoch == 0:
        # 갱신된 매개변수에 대해 훈련용 데이터의 정확도 기록
        train_acc = network.accuracy(x_train, t_train)
        train_acc_list.append(train_acc)
        # 갱신된 매개변수에 대해 검증용 데이터의 정확도 기록
        test_acc = network.accuracy(x_test, t_test)
        test_acc_list.append(test_acc)

        print(train_acc, test_acc)



