##############################################################################
# 4.4 기울기
# 수치미분
##############################################################################
import numpy as np
import matplotlib.pylab as plt


# 변수가 여럿인 함수에 대한 미분 : 편미분
# 모든 변수(x: x[0] ~ x[n])의 편미분을 벡터로 정리한 것:  기울기
#
# 신경망에서는
#   f: 손실 값 구하는 비용함수
#   x: 벡터
#       (1) 가중치(W1,W2..)의 하나의 벡터 w0, w1...
#           또는
#       (2) 편향(b1, b2..) 매개변수
def numerical_gradient_v(f, x):
    h = 1e-4                 # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):           # 은닉층 또는 출력층 수만큼
        tmp_val = x[idx]                # idx 에 해당하는 값만 기울기를 구하기 위함

        # f(x+h) 계산
        x[idx] = float(tmp_val) + h     # idx 번째 요소를 x+h 값으로 변경 -> self.W 가 변경되어 f(x) 호출된다.
        fxh1 = f(x)                     # 손실 값 계산

        # f(x-h) 계산
        x[idx] = float(tmp_val) - h     # idx 번째 요소를 x-h 값으로 변경 -> self.W 가 변경되어 f(x) 호출된다.
        fxh2 = f(x)                     # 손실 값 계산

        # 기울기 계산
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        # 값 복원 -> 매개변수들의 기울기를 구할때 해당 매개변수 외의 값은 고정되어 있어야 한다.
        x[idx] = tmp_val

    return grad

# f : 최적화하려는 함수
# X : 행렬
#       (1) 가중치 W의 형상은 (앞층의 뉴런수, 다음층의 뉴런수)
#           또는
#       (2) 편향 b의 형상은 앞층 뉴런이 없으므로 (다음층 뉴런수만,)
def numerical_gradient(f, X):
    if X.ndim == 1:
        return numerical_gradient_v(f, X)
    else:
        grad = np.zeros_like(X)

        # 각 매개변수 W1(w0, w1, w2...), W2(w0, w1, w2...) 또는 b1, b2의 기울기를 구한다.
        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient_v(f, x)

        return grad


# 변수가 두 개인(여기서는 x[0], x[1]) 함수
def function_2(x):
    if x.ndim == 1:
        return np.sum(x ** 2)   # x[0] ** 2 + x[1] ** 2
    else:
        return np.sum(x ** 2, axis=1)


# 접선 방정식 구하기
def tangent_line(f, a):
    d = numerical_gradient(f, a)
    print(d)

    return lambda x: d * x - d * a + f(a)


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)

    # 사각형 영역을 구성하는 가로축의 점들과 세로축의 점을 나타내는 두 벡터를 인수로 받아서 이 사각형 영역을 이루는 조합을 출력한다.
    # 그리드 포인트의 x 값만을 표시하는 행렬과 y 값만을 표시하는 행렬 두 개로 분리하여 출력한다.(m X n)
    X, Y = np.meshgrid(x0, x1)

    # 1차원으로 납잡하게 만듬
    X = X.flatten()
    Y = Y.flatten()

    # function_2의 기울기 구하기
    grad = numerical_gradient(function_2, np.array([X, Y]))

    plt.figure()
    # 기울기 결과에 마이너스(-) 붙인 벡터를 사용한다. -> 기울기의 반대 방향을 가리키게된다.
    #   - 각 지점에서 낮아지는 방향을 가리킨다. 즉, 기울기가 가리키는 쪽은 각 장소에서 함수의 출력 값을 가장 줄이는 방향이다.
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    # plt.show()


    #############################
    # meshgrid 를 사용해보자
    #   - 사각형 영역을 구성하는 가로축의 점들과 세로축의 점을 나타내는 두 벡터를 인수로 받아서 이 사각형 영역을 이루는 조합을 출력한다.
    #   - 그리드 포인트의 x 값만을 표시하는 행렬과 y 값만을 표시하는 행렬 두 개로 분리하여 출력한다.
    #############################
    m0 = np.arange(3)
    m1 = np.arange(5)

    X_, Y_ = np.meshgrid(m0, m1)
    print(X_)
    # [[0 1 2]
    #  [0 1 2]
    #  [0 1 2]
    #  [0 1 2]
    #  [0 1 2]]
    print(Y_)
    # [[0 0 0]
    #  [1 1 1]
    #  [2 2 2]
    #  [3 3 3]
    #  [4 4 4]]
    XY_ = [list(zip(x, y)) for x, y in zip(X_, Y_)]
    print(XY_)
    # [[(0, 0), (1, 0), (2, 0)],
    #  [(0, 1), (1, 1), (2, 1)],
    #  [(0, 2), (1, 2), (2, 2)],
    #  [(0, 3), (1, 3), (2, 3)],
    #  [(0, 4), (1, 4), (2, 4)]]

    plt.figure(2)
    plt.scatter(X_, Y_, linewidths=10)
    plt.xlabel('m0')
    plt.ylabel('m1')
    plt.show()

