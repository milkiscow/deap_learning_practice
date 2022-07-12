#넘파이 불러오기
import numpy as np
print(np.__version__)

#3x2 행렬 생성
my_arr = np.array([[10, 20, 30], [40, 50, 60]])
print(my_arr)
print(type(my_arr))
print(my_arr[0][2])
print(np.sum(my_arr))

#맷플롯립 불러오기
import matplotlib.pyplot as plt

#x 좌표와 y 좌표를 파이썬 리스트로 전달함.
plt.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]) 
plt.show()

#점그래프로 표현
plt.scatter([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])
plt.show()

#표준 정규 분포를 따르는 난수 1,000개를 만듬.
x = np.random.randn(1000) 
y = np.random.randn(1000) 
plt.scatter(x, y)
plt.show()

