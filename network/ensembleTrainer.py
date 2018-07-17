import tensorflow as tf
import time
from common.optimizer import *
from network.vgg_like.deep_convnet35_en import DeepConvNet
from collections import Counter

class Trainer:
    """신경망 훈련을 대신 해주는 클래스
    epochs : 에폭 수
    mini_batch_size : 미니배치 크기
    optimizer : 최적화 함수 이름
    optimizer_param :  최적화에 적용할 하이퍼파라미터 딕셔너리

    """

    def __init__(self, x_train, t_train, x_test, t_test
                 , epochs=20
                 , mini_batch_size=100
                 , optimizer='sgd'
                 , optimizer_param={'lr': 0.01}
                 , evaluate_sample_num_per_epoch=None
                 , verbose=True):
        self.models = [DeepConvNet(), DeepConvNet(), DeepConvNet()]
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        self.start_time = time.time()
        self.en_grads = {}
        self.en_grads2 = {}

        # optimizer
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'nesterov': Nesterov, 'adagrad': AdaGrad,
                                'rmsprpo': RMSprop, 'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        self.train_size = x_train.shape[0]
        # 1에폭이 되기위한 처리 반복 횟수
        self.iter_per_epoch = self.train_size / mini_batch_size
        # 최대 반복 수 = 원하는 에폭 수 * 1에폭당 반복 횟수
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        # trainer 결과 -> 학습시키는 동안의 loss 이력, 정확도 이력 결과
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        self.load_batch_masks = {}
        self.batch_mask_list = []
        np.random.seed(4250029868)

    def train_step(self):
        # 배치 데이터
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        self.batch_mask_list.append(batch_mask)

        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        for idx, model in enumerate(self.models):
            # 업데이트할 가중치 구하기
            grads = model.gradient(x_batch, t_batch)

            # 가중치 최적화해서 업데이트!!!
            self.optimizer.update(model.params, grads)

            # 새로 업데이트한 가중치 적용해서 손실 값 구하기
            loss = model.loss(x_batch, t_batch)
            self.train_loss_list.append(loss)

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
                train_acc = model.accuracy(x_train_sample, t_train_sample, self.batch_size)
                test_acc = model.accuracy(x_test_sample, t_test_sample, self.batch_size)
                self.train_acc_list.append(train_acc)
                self.test_acc_list.append(test_acc)

                if self.verbose:
                    end_time = time.time() - self.start_time
                    print(
                        "=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(
                            test_acc) + ", time :" + str(int(end_time)) + " seconds ===")

        # 반복횟수 저장
        self.current_iter += 1

    def train(self):
        # 최대 반복 횟수만큼 학습 시키기1
        for i in range(self.max_iter):
            self.train_step()

        # # 검증용 데이터로 정확도 확인
        # acc = 0.0
        # # predictions = np.zeros(int(self.x_test.shape[0]) * 10).reshape(int(self.x_test.shape[0]), 10)
        # # predictions = np.zeros(self.x_test.shape[0])
        #
        # predictions = np.zeros((len(self.models), self.x_test.shape[0]))
        # for idx, model in enumerate(self.models):
        #     predictions[idx] = (model.accuracy_(self.x_test, self.batch_size))
        #
        # voting_result = np.zeros(self.batch_size)
        # for idx in range(0, self.batch_size):
        #     voting_result[idx] = self.majority(predictions[:, idx])
        #
        # t = self.t_test
        # if t.ndim != 1:
        #     t = np.argmax(t, axis=1)
        #
        # acc += np.sum(voting_result == t)
        # test_acc = (acc / self.x_test.shape[0]) * 100     # 백분률 (%)



        # # 검증용 데이터로 정확도 확인
        # test_acc = 0.0
        # for idx, model in enumerate(self.models):
        #     test_acc += model.accuracy(self.x_test, self.t_test)
        #
        # test_acc = test_acc / 2

        # 검증용 데이터로 정확도 확인
        test_acc = self.ensanble_accuracy(self.x_test, self.t_test)


        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

    def ensanble_accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        predic = np.zeros((len(self.models), batch_size))
        result = np.zeros(batch_size)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            for n, model in enumerate(self.models):
                y = model.predict(tx, train_flg=False)
                predic[n] = np.argmax(y, axis=1)
            for s in range(0, batch_size):
                result[s] = self.majority(predic[:, s])
            acc += np.sum(result == tt)

        # return acc / x.shape[0]
        return (acc / x.shape[0]) * 100     # 백분률 (%)

    def majority(self, arr):
        # array 를 dictionary 로 전환
        freqDict = Counter(arr)
        # dictionary 돌면서 다수결 결과 체크
        result = (arr[0], 0)
        for (key, val) in freqDict.items():
            if result[1] < val:
                result = (key, val)

        return result[0]

if __name__ == '__main__':
    """args 여러개 전달하는 방법"""
    ##############################################################################
    # Download the fashion_mnist data
    ##############################################################################
    (x_train, t_train), (x_test, t_test) = tf.keras.datasets.fashion_mnist.load_data()

    # 시간이 오래 걸릴 경우 데이터를 줄인다.
    x_train, t_train = x_train[:2000], t_train[:2000]
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
    ##################################$#################################
    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    # print("x_train shape:", x_train.shape, "t_train shape:", t_train.shape)
    # 변경전 : x_train shape: (60000, 28, 28) t_train shape: (60000,)
    # 변경후 : x_train shape: (60000, 1, 28, 28) t_train shape: (60000,)


    trainer = Trainer(x_train, t_train, x_test, t_test
                      , epochs=1
                      , mini_batch_size=100
                      , optimizer='Adam'
                      , optimizer_param={'lr': 0.001}
                      , evaluate_sample_num_per_epoch=1000)
    trainer.train()
