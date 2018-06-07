##############################################################################
# 5.5 활성화 함수 계층
# 5.5.1 ReLU 계층
# 5.5.2 Sigmoid 계층
# 5.6.1 Affine 계층
# 5.6.3 SoftmaxWithLoss 계층

# 6.3   BatchNormalization 계층
# 6.4.3 Dropout계층
##############################################################################
from common.functions import *
from common import util

class Relu:
    def __init__(self):
        # 역전파에서 사용하기 위해 저장해둔다.
        self.mask = None

    def forward(self, x):
        # 입력데이터 x가 x <= 0인 위치를 저장해둔다.
        # bool 배열 생성,
        self.mask = (x <= 0)
        out = x.copy()
        # True인 인덱스의 값에 0 대입, 나머지는 그대로 출력
        out[self.mask] = 0

        return out

    def backward(self, dout):
        # 입력데이터 x가 x <= 0 였으면 미분값은 0, 아니면 그대로 흘린다.
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        # 역전파에서 사용하기 위해 저장해둔다.
        self.out = None

    def forward(self, x):
        # 순전파의 출력을 인스턴스 변수 out에 보관
        out = sigmoid(x)
        self.out = out

        return out

    def backward(self, dout):
        # 순전파의 출력값만으로 역전파를 구할 수 있다.
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W      # 2차원 데이터 (이전층 뉴런수 X 다음층의 뉴런수)
        self.b = b
        self.x = None
        self.dW = None  # 가중치 매개변수의 미분
        self.db = None  # 편향 매개변수의 미분
        self.original_x_shape = None

    def forward(self, x):
        # <텐서 대응 (reshape 하기 전에 형상 저장해둔다. 역전파 결과가 원래 형상과 같아야하므로)>
        self.original_x_shape = x.shape

        # 입력 데이터가 하나일때
        if x.ndim == 1:
            # 형상 변형시키기 - 예) 1차원 배열 [1 2 3 4] 형상 (4,)  -> 2차원 배열 [[1 2 3 4]] 형상 (1, 4) 로 변형
            x = x.reshape(1, x.size)
            # print("Affine 입력 데이터 형상 변형(1차원일때) : ", x.shape)
        # 입력 데이터가 하나인 경우를 처리해주시 않으면 입력 데이터 x의 형상은 대략 (4,) 형태이고
        # x.shape[0]은 4가 된다. (x.shape[1]은 에러발생)
        # 아래에서 다차원과 동일하게 처리하기위해 reshape으로 형상 변형을 해준것이다.

        # x.shape[0] 개수만큼의 묶음으로 다차원 배열의 원소 수가 변환 후에도 똑같이 유지되도록 묶어준다.
        # 1차원, 2차원의 경우 현재 형상에는 변경사항이 없다.
        # <텐서 대응 : 텐서인 경우에는 2차원 배열로 변형되어 dot 연산이 된다. => 내적하는 W가 2차원 데이터이다>
        x = x.reshape(x.shape[0], -1)
        # print("reshape : ", x, x.shape)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        # <텐서 대응 : 입력 데이터 x에 대해 원래 형상으로 변형>
        dx = dx.reshape(*self.original_x_shape)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None    # 손실
        self.y = None       # softmax의 출력
        self.t = None       # 정답 레이블(원-핫 인코딩형태)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    # '예측값 y - 정답값 t'가 SoftmaxWithLoss의 역전파 값이다.
    def backward(self, dout=1):
        batch_size = self.t.shape[0]     # 배치 데이터 수
        # 정답 레이블이 원-핫 인코딩 형태이면
        if self.t.size == self.y.size:
            # 원-핫 인코딩 형태이면 정답인덱스의 값만 1, 나머지는 0 이므로
            # 정답인덱스에서는 : 예측값 - 1
            # 나머지는       : 예측값 - 0 즉, 변경사항 없다.
            dx = (self.y - self.t) / batch_size     # todo 왜... batch_size로 나눌까?? 정규화?
        # 정답 인텍스가 담긴 형태이면
        else:
            dx = self.y.copy()
            # 첫번째 데이터 : dx[0,정답인덱스] = dx[0,정답인덱스] - 1 & 나머지는 그대로,
            # 두번째 데이터 : dx[1,정답인덱스] = dx[1,정답인덱스] - 1 & 나머지는 그대로,
            # 세번째 데이터 : dx[2,정답인덱스] = dx[2,정답인덱스] - 1 & 나머지는 그대로,
            # ...
            dx[np.arange(batch_size), self.t] -= 1  # zip 형태로 묶어 정답인덱스에 대해서만 뺄셈
            dx = dx / batch_size                    # todo 왜... batch_size로 나눌까?? 정규화?

        return dx


"""Batch Normalization 계층

    Parameters
    ----------
    gamma           : 확대
    beta            : 이동
    momentum        :
    running_mean    : 평균
    running_var     : 분산
    
"""
class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma       # 확대
        self.beta = beta         # 이동
        self.momentum = momentum
        self.input_shape = None  # 합성곱 계층은 4차원, 완전연결 계층은 2차원

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var

        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None      # 편차
        self.xn = None      # 정규화
        self.std = None     # 표준편차

        self.dgamma = None
        self.dbeta = None

    """
        x: Affine계층 출력값 
    """
    def forward(self, x, train_flag=True):
        # 원래 입력된 데이터의 형상 저장
        self.input_shape = x.shape

        # 입력값 x가 2차원이 아니면 (즉, 데이터가 1개만 입력된 경우)
        if x.ndim != 2:
            N, C, H, W = x.shape
            print("BatchNormalization forward -> N: {}, C: {}, H: {}, W: {}".format(N, C, H, W))
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flag)
        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flag):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        # 학습(훈련) 과정이면
        if train_flag:
            #
            # "열을 기준"으로 평균 (mean) 계산
            mu = x.mean(axis=0)
            # 편차 (subtract mean vector) 계산
            xc = x - mu
            # "열을 기준"으로 분산 (variance) 계산
            var = np.mean(xc ** 2, axis=0)
            # 표준편차
            std = np.sqrt(var + 10e-7)
            # 정규화
            xn = xc / std

            # backward 시에 사용할 중간 데이터 저장
            self.batch_size = x.shape[0]
            self.xc = xc    # 편차
            self.xn = xn    # 정규화
            self.std = std  # 표준편차

            # 지수이동평균을 사용 -> 테스트(검증)할 때 사용할 평균과 분산을 계산한다.
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

        # 테스트(검증) 과정이면
        else:
            # 편차
            xc = x - self.running_mean
            # 정규화 (편차 / 분산제곱)
            xn = xc / (np.sqrt(self.running_var + 10e-7))

        """
        정규화한 결과값 -> 대개 활성화 계층의 입력으로 사용된다.
        (배치 정규화 계층 앞 또는 뒤에 삽입된다.)
        """
        out = self.gamma * xn + self.beta
        return out


    def backward(self, dout):
        if dout.ndim != 2 :
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)
        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        # beta는 vector -> 열방향의 합을 구해 원래의 형상을 찾는다.
        dbeta = dout.sum(axis=0)
        # gamma도 vector -> 열방향의 합을 구해 원래의 형상을 찾는다.
        dgamma = np.sum(self.xn * dout, axis = 0)

        # 정규화 미분
        dxn = self.gamma * dout
        # 편차 미분
        dxc = dxn / self.std
        # 표준편차 미분
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        # 분산 미분
        dvar = 0.5 * dstd / self.std
        # 편차 미분 추가
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        # 평균 미분
        #dmu = np.sum(dxc, axis=0)
        #dx = dxc - dmu / self.batch_size

        dmu = -np.sum(dxc, axis=0)
        dx = dxc + dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            # dropout 비율만큼 무작위 삭제, 신호 흘리지 않는다.
            # dropout_ratio 보다 큰 값은 True로 작거나 같으면 False로 저장
            # 배치사이즈 100이면 True 또는 Fasle로 이루어진 (100, 100)
            # *x.shape = x.shape[0]
            self.mask = np.random.rand(x.shape[0]) > self.dropout_ratio
            # drop!!
            # True -> * 1,  False -> * 0
            return x * self.mask
            # [[0.         0.         0.... 0.         0.         0.3155615]
            #  [0.         0.         0.... 0.         0.         0.07640247]
            # ...
            #  [0.         0.         0.... 0.         0.         0.32959045]]
        else:
            # dropout 비율을 빼고 신호 흘린다.
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        #print("self.mask shape ->", self.mask.shape)  # (100, 100)
        return dout * self.mask

class Convolution:
    # W : 필터
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 중간 데이터(backward 시 사용)
        self.x = None
        self.col = None
        self.col_W = None

        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        # 출력 크키
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        # 입력 데이터에서 필터를 적용하는 영역별로 가로로 전개
        col = util.im2col(x, FH, FW, self.stride, self.pad)
        # 필터를 2차원 배열로 변형(reshape)하고(FN개의 묶음으로 변경) + Transpose 해서 세로로 전개 <- 필터는 im2col 할 필요 없다.
        col_W = self.W.reshape(FN, -1).T
        # < FN, C, FH, FW = 1, 2, 3, 3 일때 >
        #     (1) self.W.reshape(FN, -1) 결과
        #             [[-0.00558132 - 0.00473335  0.00669913 - 0.0099481   0.0183686 - 0.01924093
        #               - 0.00076841  0.00406638  0.00376849 - 0.00606364  0.00859418 - 0.01257911
        #               0.00217284  0.0036043   0.02057647  0.00256673  0.00883204 - 0.01496615]]
        #     (2) Transpose 결과
        #             [[-0.00558132]
        #              [-0.00473335]
        #              [0.00669913]
        #              [-0.0099481]
        #              [0.0183686]
        #              [-0.01924093]
        #              [-0.00076841]
        #              [0.00406638]
        #              [0.00376849]
        #              [-0.00606364]
        #              [0.00859418]
        #              [-0.01257911]
        #              [0.00217284]
        #              [0.0036043]
        #              [0.02057647]
        #              [0.00256673]
        #              [0.00883204]
        #              [-0.01496615]]

        out = np.dot(col, col_W) + self.b
        # 원소 수가 변환 후에도 똑같이 유지되도록 묶어준다.(-1 사용) && 축 순서를 바꿔준다.
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # reshape: [[[[2.10218838]
        #             [2.11292464]
        #             [2.1236609 ]]
        #
        #            [[2.20955099]
        #             [2.22028725]
        #             [2.23102351]]
        #
        #            [[2.31691359]
        #             [2.32764985]
        #             [2.33838611]]]]
        # out: [[[[2.10218838 2.11292464 2.1236609]
        #         [2.20955099 2.22028725 2.23102351]
        #         [2.31691359 2.32764985 2.33838611]]]]

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        # print("Convolution backward dout : ", dout)
        FN, C, FH, FW = self.W.shape
        dout = dout.tranpose(0, 2, 3, 1).reshape(-1, FN)
        # dout = dout.reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = util.col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        # backward 시 사용
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 입력 데이터에서 풀링을 적용하는 영역별로 가로로 전개
        col = util.im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # conv 계층의 입력 데이터와는 다르게 가로로 전개한 데이터를 풀링 크기(pool_h * pool_w)만큼 단위로 2차원 배열을 만든다(reshape)
        # 풀링 크기 만큼씩 크기로 원소 수를 유지하며 몇 개의 묶음으로 변형된다.
        col = col.reshape(-1, self.pool_h * self.pool_w)
        # 행(row)별 최댓값을 구한다.
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        print("arg_max  : ", self.arg_max, self.arg_max.size)

        return out

    def backward(self, dout):
        # print("Pooling backward dout : ", dout)

        # 왜??
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        # max pooling의 역젼파값 : 최대값이 속해 있는 요소의 로컬 그래디언트는 1, 나머지는 0이다.
        # dout 원소 개수 X pooling size 의 matrix 생성
        # 최대값이 아닌 원소들의 로컬 그래디언트는 0이므로 dout을 곱해도 0이 되므로 0으로 초기화시킨다.
        dmax = np.zeros((dout.size, pool_size))
        print("dmax.shape before : ", dmax.shape)

        # arg_max.size == dout.flatten size ??
        # zip 형태로 묶어, dmax의 해당 위치에 dout flatten 값을 하나씩 담는다. (최대값이 속해 있는 요소의 로컬 그래디언트는 1 * dout)
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        print("dmax.shape after : ", dmax.shape)

        # 채널별 pooling 적용 영역끼리 하나의 row로 묶는다. ???
        # 1row - 모든 데이터에대해 채널1의 폴링적용영역 전체 ?
        # 2row - 모든 데이터에대해 채널2의 폴링적용영역 전체 ?
        print("dmax.shape 0, 1, 2", dmax.shape[0], dmax.shape[1], dmax.shape[2])
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = util.col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


if __name__ == '__main__':
    X1 = np.array([[1.0, -0.5], [-2.0, 3.0]])

    """Relu 테스트"""
    # # bool 배열
    # mask = (X1 <= 0)
    # print("mask : ", mask)
    # # [[False  True]
    # #  [True False]]
    #
    # out = X1.copy()
    # # True 인 인덱스의 값에 0 대입
    # out[mask] = 0
    # print("out : ", out)
    # # [[1. 0.]
    # #  [0. 3.]]
    #
    # test_relu = Relu()  # 객체 생성
    # out = test_relu.forward(X1)
    # print(out)
    # # [[1. 0.]
    # #  [0. 3.]]

    """Affine 테스트"""
    # W = np.array([[1, 2, 3], [4, 5, 6]])
    # b = np.array([2, 2, 2])
    #
    # # 입력데이터가 1차원 & 2차원인 경우
    # test_affine = Affine(W, b)
    # test_affine.forward(np.array([1.0, -0.5]))
    # test_affine.forward(np.array([[1.0, -0.5],
    #                               [2.0, -1.0]]))
    # # 입력데이터가 텐서인 경우
    # W = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 9]])
    # b = np.array([2, 2])
    # test_affine = Affine(W, b)
    # test_affine.forward(np.array([[[1.0, -0.5, 0.],
    #                                [2.0, -1.0, 0.]],
    #                               [[3.0, -1.5, 0.],
    #                                [4.0, -2,   0.]]]))
    # # 입력 데이터 2 X 2 X 3 -> forward()에서 2 X 6 으로 변경
    # #                             [[1. - 0.5  0.   2. - 1.   0.]
    # #                              [3. - 1.5  0.   4. - 2.   0.]]


    """배치 정규화 테스트"""
    # X = np.array([[1.0, 0.5], [2.0, 3.0], [3.0, 3.0]])
    #
    # mu = X.mean(axis=0)
    # print(mu)
    # # [2.         2.16666667]
    #
    # # 편차 (subtract mean vector) 계산
    # xc = X - mu
    # print(xc)
    # # [[-1. - 1.66666667]
    # #  [0.    0.83333333]
    # #  [1.    0.83333333]]
    #
    # # "열을 기준"으로 분산 (variance) 계산
    # var = np.mean(xc ** 2, axis=0)
    # print(var)
    # # [0.66666667 1.38888889]
    #
    # # 표준편차
    # std = np.sqrt(var + 10e-7)
    # print(std)
    # # [0.81649719 1.17851173]
    #
    # # 정규화
    # xn = xc / std
    # print(xn)
    # # [[-1.22474395 - 1.41421305]
    # #  [0.            0.70710653]
    # #  [1.22474395    0.70710653]]
    #
    # """ 제곱 & 합 """
    # print(X ** 2)
    # # [[1.   0.25]
    # #  [4.   9.]
    # #  [9.   9.]]
    # print(np.sum(X ** 2))
    # # 32.25 = 1. + 0.25 + 4. + 9. + 9. + 9. (각 원소를 모두 더한 값)


    """ pooling 테스트 """
    # dout = np.array([[111, 112, 113, 114, 115, 116, 117],
    #                  [161, 162, 163, 164, 165, 166, 167],
    #                  [171, 172, 173, 174, 175, 176, 177]])
    #
    # arg_max =  np.array([[5, 7, 5, 7, 5, 7, 5],
    #                      [1, 1, 1, 1, 1, 1, 1],
    #                      [2, 2, 2, 2, 2, 2, 2]])
    # print(arg_max.size, arg_max.flatten())
    # # 21 [5 7 5 7 5 7 5 1 1 1 1 1 1 1 2 2 2 2 2 2 2]
    #
    # pool_size = 3 * 3
    # dmax = np.zeros((dout.size, pool_size))
    # print(dmax)
    # print("dmax.shape before : ", dmax.shape) # (21, 9)
    #
    # print(np.arange(arg_max.size), arg_max.flatten())
    # # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20] [5 7 5 7 5 7 5 1 1 1 1 1 1 1 2 2 2 2 2 2 2]
    # dmax[np.arange(arg_max.size), arg_max.flatten()] = dout.flatten()
    # print(dmax)
    # # zip 형태로 (0, 5), (1, 7), (2, 5) 등..의 위치에 dout flatten 값을 넣는다.
    #
    #
    #
    # # # 최대값
    # # dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
    # # dmax = dmax.reshape(dout.shape + (pool_size,))
    # # print("dmax.shape after : ", dmax.shape)



