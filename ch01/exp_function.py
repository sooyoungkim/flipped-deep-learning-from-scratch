import numpy as np
import matplotlib.pylab as plt


x = np.arange(-5.0, 5.0, 0.1)
y = np.exp(-x)


# 지수함수를 y축 대칭시킨 것과 y축으로 +1 이동 시킨 결과를 확인해보자
plt.plot(x, np.exp(x), 'r--', x, np.exp(-x), 'bs', x, 1 + np.exp(-x), 'g^')
plt.ylim(-0.1, 1.1)
plt.show()