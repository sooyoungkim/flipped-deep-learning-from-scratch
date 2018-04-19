##############################################################################
#
# 99 페이지 : MNIST 데이터를 가져와서 화면에 출력
#   - 훈련 이미지가 60,000장, 시험 이미지가 10,000장
#       => 훈련 이미지를 사용하여 모델을 학습하고,
#           학습한 모델로 시험 이미지들을 얼마나 정확하게 분류하는지를 평가한다.
#
##############################################################################
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))    # numpy array로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환
    pil_img.show()


# 데이터 가져오기 : (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
(x_train, t_train), _ = load_mnist(flatten=True, normalize=False)
print(x_train.shape)    # (60000, 784)
print(t_train.shape)    # (60000,)


# 훈련용 데이터 사용
img = x_train[0]    # training 이미지 하나 가져오기
label = t_train[0]  # training 라벨 하나 가져오기
print(label)        # 5

print(img.shape)            # (784,)
img = img.reshape(28, 28)   # 형상을 원래 이미지의 크기로 변형
print(img.shape)            # (28, 28)

img_show(img)
