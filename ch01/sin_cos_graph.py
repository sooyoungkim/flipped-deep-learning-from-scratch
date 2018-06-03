# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

##############################################################################
# 1.6.3 이미지 표시하기
# 이미지를 화면에 보여줍니다.
##############################################################################
img = imread('../dataset/lena.png') # 이미지 읽어오기
plt.imshow(img)
plt.show()

##############################################################################
# 1.6.2 pyplot의 기능
# 사인(sin) 그래프와 코사인(cos) 그래프를 그립니다.
##############################################################################
# 데이터 준비
x = np.arange(0, 6, 0.1) # 0에서 6까지 0.1 간격으로 생성
y1 = np.sin(x)
y2 = np.cos(x)

# 그래프 그리기
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label="cos") # cos 함수는 점선으로 그리기
plt.xlabel("x") # x축 이름
plt.ylabel("y") # y축 이름
plt.title('sin & cos')
plt.legend()
plt.show()
