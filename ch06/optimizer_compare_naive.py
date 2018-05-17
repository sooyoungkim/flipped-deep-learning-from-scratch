##############################################################################
# 6.1.7 어느 갱신 방법을 이용할 것인가?
# SGD, 모멘텀, AdaGrad, Adam의 학습 패턴을 비교합니다.
##############################################################################

import sys,os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *

def f(x, y):
    return x ** 2 / 20.0 + y ** 2

def f(x, y):
    return x / 10.0, 2.0 * y

init_pos = (-7.0, 2.0)
params = {}