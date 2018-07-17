##############################################################################
#
# Fashion Minit 에 LeNet 적용해서 인식률 높여보기
#
##############################################################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from network.lenet.lenet import LeNet
from common.trainer import Trainer

##############################################################################
# Download the fashion_mnist data
##############################################################################
(x_train, t_train), (x_test, t_test) = tf.keras.datasets.fashion_mnist.load_data()

# 시간이 오래 걸릴 경우 데이터를 줄인다.
# x_train, t_train = x_train[:5000], t_train[:5000]
# x_test, t_test = x_test[:1000], t_test[:1000]


##############################################################################
# Data normalization
#   Normalize the data dimensions so that they are of approximately the same scale.
#   이미지 데이터 정규화 (0 ~ 255 범위인 각 픽셀 값을 0.0 ~ 1 범위로 변환)
##############################################################################
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


###################################################################
# Fashion Mnist 데이터가 기존 Mnist와 데이터 포맷이 달라서 코드가 호환이 안된다.
# 아래 코드를 사용해서 변환하면된다
###################################################################
x_train=np.expand_dims(x_train,axis=1)
x_test=np.expand_dims(x_test,axis=1)


print("x_train shape:", x_train.shape, "t_train shape:", t_train.shape)
# 변경전 : x_train shape: (60000, 28, 28) t_train shape: (60000,)
# 변경후 : x_train shape: (60000, 1, 28, 28) t_train shape: (60000,)

network = LeNet()
trainer = Trainer(network, x_train, t_train, x_test, t_test
                  , epochs=20
                  , mini_batch_size=100
                  , optimizer='Adam'
                  , optimizer_param={'lr':0.001}
                  , evaluate_sample_num_per_epoch=1000)
trainer.train()

# # 매개변수 보관
# network.save_params("fashion-lenet_params.pkl")
# print("Saved Network Parameters!")

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
