##############################################################################
# 5.7.3 오차역전파법으로 구한 기울기 검증하기
##############################################################################
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset import mnist
from ch05 import two_layer_net

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = mnist.load_mnist(normalize=True, one_hot_label=True)

network = two_layer_net.TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

# print(x_batch)
# print(t_batch)

# 수치미분으로 구한 기울기
grad_numerical = network.numerical_gradient(x_batch, t_batch)   # grads 반환
# 역전파법으로 구한 기울기
grad_backprop = network.gradient(x_batch, t_batch)              # grads 반환

# 수치미분은 오차역전파법을 정확히 구현했는지 확인하기 위해 필요하다.
for key in grad_numerical.keys():
    # W1, b1, W2, b2 에 대해 절대값을 구한 후, 그 절대값들의 평균을 구한다.
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))
