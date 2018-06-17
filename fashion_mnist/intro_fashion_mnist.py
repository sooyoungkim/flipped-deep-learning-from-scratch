##############################################################################
#
# (1) "fashion mnist" 데이터를 로그해본다.
# (2) "mnist" 와 "fashion mnist" 데이터를 비교해보고 호환해서 사용할 수 있도록 변경해본다.
#
##############################################################################
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)
# tensorflow 버전이 낮을 경우 fashion mnist 데이터 로드시에 에러발생. (keras 사용할 수 없었다)

##############################################################################
#
# Download the "mnist" data
#
##############################################################################
(ox_train, ot_train), (ox_test, ot_test) = load_mnist(normalize=True, flatten=False, one_hot_label=False)

# 형상 확인 : ndim = 4
print("x_train shape:", ox_train.shape, "t_train shape:", ot_train.shape)
# x_train shape: (60000, 1, 28, 28) t_train shape: (60000,)

# training and test datasets 데이터 수 확인
print(ox_train.shape[0], 'train set')
print(ox_test.shape[0], 'test set')
# 60000 train set
# 10000 test set

# 데이터 하나 확인해보자
# print(ox_train[0])
# 데이터 타입
print(type(ox_train[0]))             # <class 'numpy.ndarray'>
print(type(ox_train[0][0]))          # <class 'numpy.ndarray'>
print(type(ox_train[0][0][0]))       # <class 'numpy.ndarray'>
print(type(ox_train[0][0][0][0]))    # <class 'numpy.float32'>


##############################################################################
# Download the "fashion mnist" data
#
# mnist와는
#   (1) 형상이 다르다.
#   (2) 엘리멘트의 타입이 다르다.
#   (3) mnist와 fashion mnist 모두 정규화해서 사용하도록한다.
#
##############################################################################
(x_train, t_train), (x_test, t_test) = tf.keras.datasets.fashion_mnist.load_data()

# 형상 확인 : ndim = 3
#   there are 60,000 training data of image size of 28x28, 60,000 train labels)
print("x_train shape:", x_train.shape, "t_train shape:", t_train.shape)
# x_train shape: (60000, 28, 28) t_train shape: (60000,)

# training and test datasets 데이터 수 확인
print(x_train.shape[0], 'train set')
print(x_test.shape[0], 'test set')
# 60000 train set
# 10000 test set

# 데이터 하나 확인해보자
# print(x_train[0])
# 데이터 타입
print(type(x_train[0]))         # <class 'numpy.ndarray'>
print(type(x_train[0][0]))      # <class 'numpy.ndarray'>
print(type(x_train[0][0][0]))   # <class 'numpy.uint8'>     -> out range : 0 to 255 (Unsigned 8-bit integer)


##############################################################################
# Visualize the data
##############################################################################
# Define the text labels
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2
                        "Dress",        # index 3
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6
                        "Sneaker",      # index 7
                        "Bag",          # index 8
                        "Ankle boot"]   # index 9

# Image index, you can pick any number between 0 and 59,999
img_index = 5
# y_train contains the lables, ranging from 0 to 9
label_index = t_train[img_index]
# Print the label, for example 2 Pullover
print ("y = " + str(label_index) + " " +(fashion_mnist_labels[label_index]))
# # Show one of the images from the training dataset
plt.interactive(False)
plt.imshow(x_train[img_index])
plt.show()


plt.imshow(x_train[1, :], cmap='gray')
plt.show()

###################################################################
# Fashion Mnist 데이터가 기존 Mnist와 데이터와 형상이 달라서 코드가 호환이 안된다.
# 아래 코드를 사용해서 변환하면된다
###################################################################
x_train=np.expand_dims(x_train,axis=1)
x_test=np.expand_dims(x_test,axis=1)

##############################################################################
# Data normalization
#   Normalize the data dimensions so that they are of approximately the same scale.
#   이미지 데이터 정규화 (0 ~ 255 범위인 각 픽셀 값을 0.0 ~ 1.0 범위로 변환)
#       엘리멘트의 타입이 다르다. mnist는 float32 타입이고 fashion mnist는 uint8
#           그래서 fashion mnist 데이터를 바로 255 로 나누면 모두 0이 되어버린다.(Unsigned integer 이므로)
#               타입변환을 해주고 255로 나눈다.
##############################################################################
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


