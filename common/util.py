import numpy as np


def smooth_curve(x):
    """손실 함수의 그래프를 매끄럽게 하기 위해 사용

    참고：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[5:len(y) - 5]


def shuffle_dataset(x, t):
    """데이터셋을 뒤섞는다.
    Parameters
    ----------
    x : 훈련 데이터
    t : 정답 레이블

    Returns
    -------
    x, t : 뒤섞은 훈련 데이터와 정답 레이블
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    t = t[permutation]

    return x, t


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 4차원 이미지(데이터)를 입력받아 2차원 배열로 변환한다(평탄화).
       합성곱에서 필터링(가중치 계산)하기 좋게 전개하는 함수이다.

    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(데이터 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    col : 2차원 배열
    """
    # 데이터 수, 채널 수, 높이(행), 너비(열)
    N, C, H, W = input_data.shape
    # 출력크기 계산
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 패딩 추가하기
    # < pad_width > : Number of padded to the deges of each axis. [(before_1st, after_1st)..(before_N, after_N)]
    # npadp[0] = (0,0) : 첫번째 축을 기준으로 앞에 padding 0개, 뒤에 padding 0개를 붙인다.
    # npadp[1] = (0,0) : 두번째 축을 기준으로 위에 padding 0개, 아래에 padding 0개를 붙인다.
    # npadp[2] = (pad,pad) : 세번째 축을 기준으로 위에 padding pad개, 아래에 padding pad개를 붙인다.
    # npadp[3] = (pad,pad) : 네번째 축을 기준으로 위에 padding pad개, 아래에 padding pad개를 붙인다.
    npad = [(0, 0), (0, 0), (pad, pad), (pad, pad)]
    img = np.pad(input_data, npad, 'constant')

    # 0으로 채워진  N x C x FH x FW x OH x OW 형상 배열 생성
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 입력 데이터(input_data)에 대해 한번에 모두 꺼내서 메모리에 올려놓고 행렬 계산에 사용할 수 있다.
    for y in range(filter_h):
        # 현재 y에서 height의 위치 최대값 ( 현재위치 + 다음위치[스트라이드 * 출력 height] )
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            # print("(y, x)=(", y, ",", x, ")", "(y_max, x_max)=(", y_max, ",", x_max, ")")

            # slice -> [start : end : step]
            # print("default : ", col[:, :, y, x, :, :])
            # col[:, :, y, x, :, :]
            #       -> 전체, 전체, y번째(for), x번째(for), 전체, 전체
            #       데이터전체(N), 채널전체(C)에서 -> "각 y번째 추려내기 -> 각 y번째 안에서 각 x번째 추려내기" -> 그 안의 데이터
            #       (N, C, out_h, out_w) -> (1, 2, 3, 3)
            #        예) 채널이 2개라면 Channel[0]의 y[2], x[1] 안의 데이터 & Channel[1]의 y[2], x[1] 안의 데이터
            #
            #         [   # 1(N)
            #             [   # 2(C)
            #                 [   # 2(out_h), 2(out_w)
            #                     [0. 0. 0.]
            #                     [0. 0. 0.]
            #                     [0. 0. 0.]]
            #
            #                 [
            #                     [0. 0. 0.]
            #                     [0. 0. 0.]
            #                     [0. 0. 0.]]
            #             ]
            #         ]

            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
            # print("img : ", img[:, :, y:y_max:stride, x:x_max:stride])
            # img[:, :, y:y_max:stride, x:x_max:stride]
            #       -> 전체, 전체, y ~ (y_max -1) 까지 slide 만큼씩, x ~ (x_max -1) 까지 slide 만큼씩

        # print("before : ", col)

    # 축의 위치 변경 & 원소 수 유지하면서 2차원 배열로 변형 (N * out_h * out_w 개의 묶음으로 만든다.)
    # print("축 변경 : ", col.transpose(0, 4, 5, 1, 2, 3))
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    # print("after : ", col)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.

    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


if __name__ == '__main__':
    # 4차원 데이터 생성
    # 데이터 수, 채널 수, 높이, 너비
    # X1 = np.random.rand(1, 3, 7, 7)
    # [
    #   [
    #      [[0.72880098   0.32546482  0.62549463  0.21464958  0.67260375  0.8630507   0.39157925]
    #       [0.99926683   0.77267306  0.58687145  0.40921468  0.11554076  0.46180425  0.1155739]
    #       [0.53124025   0.79038418  0.29129996  0.90897402  0.1411847   0.21172184  0.380683]
    #       [0.47562996   0.02886909  0.89915145  0.97528882  0.27963553  0.65716417  0.76268245]
    #       [0.52371597   0.29620359  0.27636053  0.75774751  0.75122158  0.6264706   0.15564784]
    #       [0.41187152   0.54298476  0.41735015  0.03430846  0.55622194  0.66341337  0.26868117]
    #       [0.28355161   0.02226755  0.1146015   0.7664987   0.04383611  0.22217184  0.45936275]]
    #
    #       [[0.72880098   0.32546482  0.62549463  0.21464958  0.67260375  0.8630507   0.39157925]
    #        [0.99926683   0.77267306  0.58687145  0.40921468  0.11554076  0.46180425  0.1155739]
    #        [0.53124025   0.79038418  0.29129996  0.90897402  0.1411847   0.21172184  0.380683]
    #        [0.47562996   0.02886909  0.89915145  0.97528882  0.27963553  0.65716417  0.76268245]
    #        [0.52371597   0.29620359  0.27636053  0.75774751  0.75122158  0.6264706   0.15564784]
    #        [0.41187152   0.54298476  0.41735015  0.03430846  0.55622194  0.66341337  0.26868117]
    #        [0.28355161   0.02226755  0.1146015   0.7664987   0.04383611  0.22217184  0.45936275]]
    #
    #       [[0.72880098   0.32546482  0.62549463  0.21464958  0.67260375  0.8630507   0.39157925]
    #        [0.99926683   0.77267306  0.58687145  0.40921468  0.11554076  0.46180425  0.1155739]
    #        [0.53124025   0.79038418  0.29129996  0.90897402  0.1411847   0.21172184  0.380683]
    #        [0.47562996   0.02886909  0.89915145  0.97528882  0.27963553  0.65716417  0.76268245]
    #        [0.52371597   0.29620359  0.27636053  0.75774751  0.75122158  0.6264706   0.15564784]
    #        [0.41187152   0.54298476  0.41735015  0.03430846  0.55622194  0.66341337  0.26868117]
    #        [0.28355161   0.02226755  0.1146015   0.7664987   0.04383611  0.22217184  0.45936275]]
    #   ]
    # ]

    # 패딩 추가 1.
    # Z = np.ones((5,5))
    # Z = np.pad(Z, pad_width=1, mode="constant", constant_values=0)
    # print(Z)
    # [[0. 0. 0. 0. 0. 0. 0.]
    #  [0. 1. 1. 1. 1. 1. 0.]
    #  [0. 1. 1. 1. 1. 1. 0.]
    #  [0. 1. 1. 1. 1. 1. 0.]
    #  [0. 1. 1. 1. 1. 1. 0.]
    #  [0. 1. 1. 1. 1. 1. 0.]
    #  [0. 0. 0. 0. 0. 0. 0.]]

    # 패딩 추가 2.
    # X2 = np.random.rand(1, 1, 3, 3)
    # print(X2)
    # print(np.pad(X2, [(0, 0), (0, 0), (3, 3), (3, 3)], 'constant'))
    #
    # [[[[0.91347297 0.89350636 0.01989363]
    #    [0.52972723 0.35484639 0.75199472]
    #    [0.16447222 0.56245324 0.68867721]]]]
    #
    # [[[[0.  0.  0.  0.          0.          0.          0.  0.  0.]
    #    [0.  0.  0.  0.          0.          0.          0.  0.  0.]
    #    [0.  0.  0.  0.          0.          0.          0.  0.  0.]
    #    [0.  0.  0.  0.91347297  0.89350636  0.01989363  0.  0.  0.]
    #    [0.  0.  0.  0.52972723  0.35484639  0.75199472  0.  0.  0.]
    #    [0.  0.  0.  0.16447222  0.56245324  0.68867721  0.  0.  0.]
    #    [0.  0.  0.  0.          0.          0.          0.  0.  0.]
    #    [0.  0.  0.  0.          0.          0.          0.  0.  0.]
    #    [0.  0.  0.  0.          0.          0.          0.  0.  0.]]]]

    # 패딩 추가 3. 축별로 어떻게 패딩이 적용되는가?
    # X2 = np.random.rand(1, 1, 3, 3)
    # print(X2)
    # print(np.pad(X2, [(0, 0), (1, 2), (0, 0), (0, 0)], 'constant'))
    #
    # np.pad(X2, [(0, 0), (0, 0), (0, 0), (1, 2)]) 일때
    # [
    #     [
    #         [
    #             [0.         0.49794697 0.21681608 0.52198091    0.         0.]
    #             [0.         0.52181368 0.55996258 0.26520515    0.         0.]
    #             [0.         0.83388949 0.40886887 0.5765385     0.         0.]
    #         ]
    #     ]
    # ]
    # np.pad(X2, [(0, 0), (0, 0), (1, 2), (0, 0)]) 일때
    # [
    #     [
    #         [
    #             [0.         0.         0.        ]
    #             [0.67966789 0.82249291 0.28106467]
    #             [0.57310291 0.2506807  0.65650287]
    #             [0.23060743 0.72615059 0.28383698]
    #             [0.         0.         0.        ]
    #             [0.         0.         0.        ]
    #         ]
    #     ]
    # ]
    # np.pad(X2, [(0, 0), (1, 1), (0, 0), (0, 0)])
    # [
    #     [
    #         [
    #             [0.         0.         0.]
    #             [0.         0.         0.]
    #             [0.         0.         0.]
    #         ]
    #         [
    #             [0.42138947 0.01685031 0.76361683]
    #             [0.21743397 0.71135289 0.68898746]
    #             [0.53867965 0.81375595 0.57046507]
    #         ]
    #         [
    #             [0.         0.         0.]
    #             [0.         0.         0.]
    #             [0.         0.         0.]
    #         ]
    #         [
    #             [0.         0.         0.]
    #             [0.         0.         0.]
    #             [0.         0.         0.]
    #         ]
    #     ]
    # ]

    # im2col 호출
    x = np.array([[[[111, 112, 113, 114, 115, 116, 117],
                    [121, 122, 123, 124, 125, 126, 127],
                    [131, 132, 133, 134, 135, 136, 137],
                    [141, 142, 143, 144, 145, 146, 147],
                    [151, 152, 153, 154, 155, 156, 157],
                    [161, 162, 163, 164, 165, 166, 167],
                    [171, 172, 173, 174, 175, 176, 177]],
                   [[211, 212, 213, 214, 215, 216, 217],
                    [221, 222, 223, 224, 225, 226, 227],
                    [231, 232, 233, 234, 235, 236, 237],
                    [241, 242, 243, 244, 245, 246, 247],
                    [251, 252, 253, 254, 255, 256, 257],
                    [261, 262, 263, 264, 265, 266, 267],
                    [271, 272, 273, 274, 275, 276, 277]]
                   ]])
    # im2col(x, 3, 3, stride=2)
