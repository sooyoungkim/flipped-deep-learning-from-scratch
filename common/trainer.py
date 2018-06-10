import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.optimizer import *

class Trainer:
    """신경망 훈련을 대신 해주는 클래스
    epochs : 에폭 수
    mini_batch_size : 미니배치 크기
    optimizer : 최적화 함수 이름
    optimizer_param :  최적화에 적용할 하이퍼파라미터 딕셔너리

    """
    def __init__(self, network, x_train, t_train, x_test, t_test
                 , epochs=20
                 , mini_batch_size=100
                 , optimizer='sgd'
                 , optimizer_param={'lr': 0.01}
                 , evaluate_sample_num_per_epoch=None
                 , verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimizer
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'nesterov': Nesterov, 'adagrad': AdaGrad, 'rmsprpo': RMSprop, 'adam': Adam}
        # optimizer 하이퍼파라미터 설정
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        # 1에폭이 되기위한 처리 반복 횟수
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        # 최대 반복 수 = 원하는 에폭 수 * 1에폭당 반복 횟수
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        # trainder 결과 -> 학습시키는 동안의 loss 이력, 정확도 이력 결과
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        # 배치 데이터
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # 업데이트할 가중치 구하기 -> backward
        grads = self.network.gradient(x_batch, t_batch)
        # print(grads.keys())
        # 가중치 최적화해서 업데이트!!!
        self.optimizer.update(self.network.params, grads)

        # 새로 업데이트한 가중치 적용해서 손실 값 구하기 -> forward
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print("train loss:" + str(loss))

        # 1 에폭당 샘플데이터로 정확도 이력 저장
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            # 평가 샘플데이터 수를 정해준 경우
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            # 학습시마다의 정확도 이력 저장 -> pyplot 사용해 그래프로 훈련과 검증용 데이터의 정확도 비교 가능
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print(
                "=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")

        # 반복횟수 저장
        self.current_iter += 1

    def train(self):
        # 최대 반복 횟수만큼 학습 시키기
        for i in range(self.max_iter):
            self.train_step()

        # 검증용 데이터로 정확도 확인
        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))


if __name__ == '__main__':
    """args 여러개 전달하는 방법"""
    # optimizer_param={'lr': 0.02}
    # Momentum(**optimizer_param)  # Momentum의 self.lr = 0.02, self.momentum = 0.9로 초기화
    # optimizer_param = {'lr': 0.05, 'momentum': 0.7}
    # Momentum(**optimizer_param)  # Momentum의 self.lr = 0.05, self.momentum = 0.7로 초기화
