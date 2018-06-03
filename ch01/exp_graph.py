import numpy as np
import matplotlib.pylab as plt


x = np.arange(-5.0, 5.0, 0.1)
y = np.exp(-x)


# (1) 지수함수 (2) y축 대칭시킨 지수함수 (3) y축 대칭 && y축으로 +1 이동 시킨 결과를 확인해보자
plt.plot(x, np.exp(x), 'r--',
         x, np.exp(-x), 'bs',
         x, np.exp(-x) + 1, 'g^')
plt.ylim(-0.1, 1.1)
plt.show()
