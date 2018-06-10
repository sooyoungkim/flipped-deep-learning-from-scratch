import tensorflow as tf
import numpy as np
from ch08 import deep_convnet2
from common.trainer import Trainer

##############################################################################
# Download the fashion_mnist data
##############################################################################
(x_train, t_train), (x_test, t_test) = tf.keras.datasets.fashion_mnist.load_data()

# 시간이 오래 걸릴 경우 데이터를 줄인다.
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

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
x_train=np.expand_dims(x_train, axis=1)
x_test=np.expand_dims(x_test, axis=1)

print("x_train shape:", x_train.shape, "t_train shape:", t_train.shape)
# 변경전 : x_train shape: (60000, 28, 28) t_train shape: (60000,)
# 변경후 : x_train shape: (60000, 1, 28, 28) t_train shape: (60000,)


# 시간이 오래 걸릴 경우 데이터를 줄인다.
# x_train, t_train = x_train[:5000], t_train[:5000]
# x_test, t_test = x_test[:1000], t_test[:1000]

network = deep_convnet2.DeepConvNet()
trainer = Trainer(network, x_train, t_train, x_test, t_test
                  , epochs=20
                  , mini_batch_size=100
                  , optimizer='Adam'
                  , optimizer_param={'lr':0.001}
                  , evaluate_sample_num_per_epoch=1000)
trainer.train()

# 매개변수 보관
network.save_params("fashion_deep_conv_params.pkl")
print("Saved Network Parameters!")

# =============== Final Test Accuracy ===============
# test acc:86.7
# Saved Network Parameters!
