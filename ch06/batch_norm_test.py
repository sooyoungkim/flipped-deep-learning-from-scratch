##############################################################################
# 6.3.2 배치 정규화의 효과
# MNIST 데이터셋 학습에 배치 정규화를 적용해봅니다
##############################################################################
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common import multi_layer_net_extend
from common import optimizer


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 학습 데이터를 줄임
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 10
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

"""
입력 784개 
은닉층 5개
10개 출력(분류 10개)
가중치의 표준편차를 입력받은 값으로 설정 
배치 정규화 사용으로 설정
"""
def __train(weight_init_std):
    normal_network = multi_layer_net_extend.MultiLayerNetExtend(input_size = 784
                                     , hidden_size_list=[100, 100, 100, 100, 100]
                                     , output_size = 10
                                     , weight_init_std=weight_init_std)
    bach_norm_network = multi_layer_net_extend.MultiLayerNetExtend(input_size = 784
                                     , hidden_size_list=[100, 100, 100, 100, 100]
                                     , output_size = 10
                                     , weight_init_std=weight_init_std
                                     , use_batchnorm=True)
    # <>
    optimizer_ = optimizer.SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    # 에폭당 반복횟수 : iter_per_epoch 만큼 반복해야 1에폭
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    # 10억번
    for i in range(100):
    # for i in range(1000000000):
        # 훈련데이터 중 일부를 무작위로 추출하여 사용 - 미니배치 단위
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # bach_norm_network 과 normal_network 을 한번씩 실행
        for network in (bach_norm_network, normal_network):
            # 기울기 구하기
            grads = network.gradient(x_batch, t_batch)
            # 기울기 최적화해서 업데이트 하기
            optimizer_.update(network.params, grads)

        # 1에폭당
        # 업데이트된 기울기로 훈련시 정확도 측정
        if i % iter_per_epoch == 0:
            train_acc = normal_network.accuracy(x_train, t_train)
            bn_train_acc = bach_norm_network.accuracy(x_train, t_train)

            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs: break

    print("epoch_cnt : ", epoch_cnt)
    # 훈련 결과 리턴
    return train_acc_list, bn_train_acc_list


# 그래프 그리기
weight_scale_list = np.logspace(0, -4, num=4)  # log0부터 시작 log값 4개 생성
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):   # i : index, w : log value (가중치 표준편차로 사용)
    print("============== " + str(i + 1) + "/4" + " ==============")
    train_acc_list, bn_train_acc_list = __train(w)

    plt.subplot(2, 2, i + 1)
    plt.title("W:" + str(w))
    if i == 3:
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", label='Normal(without BatchNorm)', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)

    plt.ylim(0, 1.0)
    if i % 2:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 2:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    plt.legend(loc='lower right')

plt.show()
