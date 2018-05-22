##############################################################################
# 6.4.3 드롭아웃
# 일부러 오버피팅을 일으킨 후 드롭아웃(dropout)의 효과를 관찰합니다.
##############################################################################
import os
import sys

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common import multi_layer_net_extend
from common import trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

# 드롭아웃 사용 유무와 비울 설정 ============================
# 드롭아웃을 쓰지 않을 때는 False
use_dropout = True
dropout_ratio = 0.15
# ====================================================
network = multi_layer_net_extend.MultiLayerNetExtend(input_size=784
                              , hidden_size_list=[100, 100, 100, 100, 100, 100]
                              , output_size=10
                              , use_dropout=use_dropout
                              , dropout_ration=dropout_ratio)
trainer = trainer.Trainer(network, x_train, t_train, x_test, t_test
                          , epochs=31
                          , mini_batch_size=100
                          , optimizer='sgd'
                          , optimizer_param={'lr': 0.01}
                          , verbose=False)


trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()