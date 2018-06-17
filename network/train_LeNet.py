
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist
from network.lenet2 import LeNet
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

print("x_train shape:", x_train.shape, "t_train shape:", t_train.shape)
# x_train shape: (60000, 1, 28, 28) t_train shape: (60000,)


# 시간이 오래 걸릴 경우 데이터를 줄인다.
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]


network = LeNet()
trainer = Trainer(network, x_train, t_train, x_test, t_test
                  , epochs=10
                  , mini_batch_size=1000
                  , optimizer='Adam'
                  , optimizer_param={'lr':0.001}
                  , evaluate_sample_num_per_epoch=1000)
trainer.train()

# 매개변수 보관
network.save_params("lenet_params.pkl")
print("Saved Network Parameters!")