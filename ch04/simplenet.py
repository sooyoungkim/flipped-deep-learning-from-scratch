##############################################################################
# 4.4.2 신경망에서의 기울기
#   가중치 매개변수에 대한 손실함수의 기울기를 구한다.
#   가중치 W를 조금 변경했을때 손실함수 L이 얼마나 변화하느냐를 나타낸다.
##############################################################################
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common import functions
from ch04 import gradient


class simpleNet:
    def __init__(self):
        # 초기화 : 형상이 2 X 3 인 가중치 매개변수
        self.W = np.random.randn(2, 3)   # 정규분포로 초기화

    # 예측 수행
    def predict(self, x):
        a = np.dot(x, self.W)       # 신경망 각 층의 계산은 행렬의 내젹으로 처리할 수 있다
        y = functions.softmax(a)    # 출력층 활성화함수로 소프트맥스함수 사용
        return y

    # 손실 구하기
    def loss(self, x, t):           # x: 입력 매개변수 데이터, t: 정답 레이블
        #print("입력 : {}, 정답 : {}".format(x, t))
        print("self.W : {}".format(self.W))
        y = self.predict(x)                         # 입력 매개변수 x(x0 ~ xn)에 대한 예측 값 구하기
        loss = functions.cross_entropy_error(y, t)  # 교차 엔트로피오차를 손실함수로 사용 - 예측 값과 정답 레이블을 사용하여 오차 측정

        return loss                                 # 예측 값과 정답의 차이가 클 수록 손실값은 더 커진다. 즉, loss 값이 더 작으면 정답일 가능성이 높다고 판단된다


net = simpleNet()
print("초기 가중치 W : {}".format(net.W))


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

# 인수 w는 더미 -> 사용안됨, 기울기 구할때 f(x) 형태를 유지하기 위한 용도
f = lambda w: net.loss(x, t)    # 입력데이터, 정답레이블은 동일하게 유지되면서 매개변수만 갱신된다.

# 인수 w는 더미 -> 사용안됨, 기울기 구할때 f(x) 형태를 유지하기 위한 용도
def f2(W):
    return net.loss(x, t)       # 입력데이터, 정답레이블은 동일하게 유지되면서 매개변수만 갱신된다.

# 신경망의 기울기 구하기
# 각 가중치 매개변수에 대한 손실 함수의 기울기를 구한다.
#   -> 신경망의 기울기를 구한 다음에는 경사법에 따라 가중치 매개변수를 갱신하기만 하면 된다.
#   -> 손실 함수값을 줄이는 것이 목표이다.
dW = gradient.numerical_gradient(f, net.W)      # net.W 가 갱신된 값을 돌려받는다.

#dW = gradient.numerical_gradient(f2, net.W)     # net.W 가 갱신된 값을 돌려받는다.
print("계산된 가중치 W : {}".format(dW))
print("가중치 W : {}".format(net.W))   # 아직 갱신하지 않았으므로 net.W는 그대로이다.
