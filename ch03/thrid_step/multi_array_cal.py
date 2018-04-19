import numpy as np


#####################
# 다차원 배열의 차원, 형상
#####################
A = np.array([1, 2, 3, 4])
print(A)            # [1 2 3 4]
print(np.ndim(A))   # 1
print(A.shape)      # (4,)  <--- 1차원 배열이라도 다차원 배열일 때와 통일된 형태로 결과를 반환하기 위해 튜플 값으로 반환된다.
print(A.shape[0])   # 4     <--- 그냥 4!!!

A = A.reshape(1, A.size)  # <--- A 배열 형상 변형시키기
print(A)            # [[1 2 3 4]]
print(np.ndim(A))   # 2
print(A.shape)      # (1, 4)
print(A.shape[0])   # 1    <--- 1 X 4 배열의 1


B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
# [[1 2]
#  [3 4]
#  [5 6]]
print(np.ndim(B))   # 2
print(B.shape)      # (3, 2)
print(B.shape[0])   # 3     <--- 3 X 2 배열의 3


C = np.array([[2, 3], [0, 9]])
print(C)
# [[2 3]
#  [0 9]]
print(np.ndim(C))   # 2
print(C.shape)      # (2, 2)
print(C.shape[0])   # 2     <--- 2 X 2 배열의 2


##################
# 행렬의 내적(행렬 곱)
##################
D = np.dot(B, C)    # (3, 2) X (2, 2)
print(D)
# [[ 2 21]
#  [ 6 45]
#  [10 69]]
print(np.ndim(D))   # 2
print(D.shape)      # (3, 2)


# 에러발생 :
#   ValueError: shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)
#E = np.dot(np.array([[1, 2, 3], [4, 5, 6]]), C)
