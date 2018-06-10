import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)

##############################################################################
# Download the fashion_mnist data
##############################################################################
(x_train, t_train), (x_test, t_test) = tf.keras.datasets.fashion_mnist.load_data()

# Print training set shape - note there are 60,000 training data of image size of 28x28, 60,000 train labels)
print("x_train shape:", x_train.shape, "t_train shape:", t_train.shape)
# x_train shape: (60000, 28, 28) t_train shape: (60000,)

# Print the number of training and test datasets
print(x_train.shape[0], 'train set')
print(x_test.shape[0], 'test set')
# 60000 train set
# 10000 test set

##############################################################################
# Data normalization
#   Normalize the data dimensions so that they are of approximately the same scale.
#   이미지 데이터 정규화 (0 ~ 255 범위인 각 픽셀 값을 0.0 ~ 1 범위로 변환)
##############################################################################
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


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
