import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist
from ch08.deep_convnet import DeepConvNet
from common.trainer import Trainer
import numpy as np
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]


network = DeepConvNet()
trainer = Trainer(network, x_train, t_train, x_test, t_test
                  , epochs=1
                  , mini_batch_size=100
                  , optimizer='Adam'
                  , optimizer_param={'lr':0.001}
                  , evaluate_sample_num_per_epoch=1000)
trainer.train()

# # 매개변수 보관
# network.save_params("deep_convnet_params.pkl")
# print("Saved Network Parameters!")

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=1)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=1)
plt.xlabel("epochs")
plt.ylabel("accuracy(%)")
# plt.ylim(0, 1.0)
plt.ylim(0, 100.0)
plt.legend(loc='lower right')
plt.show()