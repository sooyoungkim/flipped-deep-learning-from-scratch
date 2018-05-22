##############################################################################
# 6.4.2 가중치 감소
# 일부러 오버피팅을 일으킨 후 가중치 감소(weight_decay)의 효과를 관찰합니다.
##############################################################################
import os
import sys

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common import multi_layer_net_extend
from common import optimizer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

# weight decay（가중치 감쇠） 설정 ==============================
# weight decay를 사용하지 않을 경우 0 과 동일
weight_decay_lambda = 0.1
# ==========================================================

network = multi_layer_net_extend.MultiLayerNetExtend(input_size=784
                                               , hidden_size_list=[100, 100, 100, 100, 100, 100]
                                               , output_size=10
                                               , weight_decay_lambda=weight_decay_lambda)

# 학습률이 0.01인 SGD로 매개변수
optimizer = optimizer.SGD(lr=0.01)

max_epochs = 201
# max_epochs = 10
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 에폭당 반복횟수 : iter_per_epoch 만큼 반복해야 1에폭
iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

# 10억번 반복
for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 구하기 (forward -> backward -> 기울기 구하기)
    grads = network.gradient(x_batch, t_batch)
    # 더 최적화된 매개변수 값으로 업데이트
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch: " + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()