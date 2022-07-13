#넘파이 불러오기
import numpy as np
print(np.__version__)
#맷플롯립 불러오기
import matplotlib.pyplot as plt

#Day2 선형회귀
#사이킷런에서 당뇨병 환자 데이터 가져오기
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
print(diabetes.data.shape, diabetes.target.shape)
diabetes.data[0:3]
diabetes.target[:3]

#당뇨병 환자의 데이터로 산점도 그리기
plt.scatter(diabetes.data[:, 2], diabetes.target)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#당뇨병 환자의 데이터를 입력과 타깃, 즉 x, y로 설정
x = diabetes.data[:, 2]
y = diabetes.target

#경사하강법 w와 b 초기화하기 (y_hat = wx + b)
w = 1.0
b = 1.0

#첫 번째 훈련으로 샘플 데이터 y_hat 얻기 및 타깃과 비교
y_hat = x[0] * w + b
print(y_hat)
print(y[0])

#가중치 조절
w_inc = w + 0.1
y_hat_inc = w_inc * x[0] + b
print(y_hat_inc)

#가중치 변화율
w_rate = (y_hat_inc - y_hat) / (w_inc - w)
print(w_rate)

#가중치 업데이트
w_new = w + w_rate
print(w_new)

#절편 조절
b_inc = b + 0.1
y_hat_inc = x[0] * w + b_inc
print(y_hat_inc)

#절편의 변화율
b_rate = (y_hat_inc - y_hat) / (b_inc - b)
print(b_rate)

#절편 업데이트
b_new = b + 1
print(b_new)

#오차 역전파 (y_hat과 y의 차이를 이용하여 w와 b를 업데이트)
err = y[0] - y_hat
w_new = w + w_rate * err
b_new = b + 1 * err
print(w_new, b_new)

#오차 업데이트
y_hat = x[1] * w_new + b_new
err = y[1] - y_hat
w_rate = x[1]
w_new = w_new + w_rate * err
b_new = b_new + 1 * err
print(w_new, b_new)

#전체 샘플 반복하기
for x_i, y_i in zip(x, y):
    y_hat = x_i * w + b
    err = y_i - y_hat
    w_rate = x_i
    w = w + w_rate * err
    b = b + 1 * err
print(w, b)

#샘플 확인하기
plt.scatter(x, y)
pt1 = (-0.1, -0.1 * w + b)
pt2 = (0.15, 0.15 * w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#여러 에포크 반복하기(정밀한 계산)
for i in range(1, 100):
    for x_i, y_i in zip(x, y):
        y_hat = x_i * w + b
        err = y_i - y_hat
        w_rate = x_i
        w = w + w_rate * err
        b = b + 1 * err
print(w, b)

#샘플 확인하기
plt.scatter(x, y)
pt1 = (-0.1, -0.1 * w + b)
pt2 = (0.15, 0.15 * w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#모델로 예측해보기
x_new = 0.18
y_pred = x_new * w + b
print(y_pred)

#데이터를 모델 위에서 확인하기
plt.scatter(x, y)
plt.scatter(x_new, y_pred)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#가충치에 대하여 제곱 오차 미분하기
y_hat = x_i * w + b
err = y_i - y_hat
w_rate = x_i
w = w + w_rate * err

#절편에 대하여 제곱 오차 미분하기
err = y_i - y_hat
b = b + 1 * err

#뉴런 만들기
class Neuron:
    
    def __init__(self):
        self.w = 1.0     # 가중치를 초기화합니다
        self.b = 1.0     # 절편을 초기화합니다
    
    def forpass(self, x):       #정방향 계산
        y_hat = x * self.w + self.b       # 직선 방정식을 계산합니다
        return y_hat
    
    def backprop(self, x, err):     #역방향 계산
        w_grad = x * err    # 가중치에 대한 그래디언트를 계산합니다
        b_grad = 1 * err    # 절편에 대한 그래디언트를 계산합니다
        return w_grad, b_grad

    def fit(self, x, y, epochs=100):        #모델 훈련
        for i in range(epochs):           # 에포크만큼 반복합니다
            for x_i, y_i in zip(x, y):    # 모든 샘플에 대해 반복합니다
                y_hat = self.forpass(x_i) # 정방향 계산
                err = -(y_i - y_hat)      # 오차 계산
                w_grad, b_grad = self.backprop(x_i, err)  # 역방향 계산
                self.w -= w_grad          # 가중치 업데이트
                self.b -= b_grad          # 절편 업데이트
    
#훈련해보기            
neuron = Neuron()
neuron.fit(x, y)        #자동으로 100번 반복함.

#학습된 모델의 가중치와 절편 확인하기
plt.scatter(x, y)
pt1 = (-0.1, -0.1 * neuron.w + neuron.b)
pt2 = (0.15, 0.15 * neuron.w + neuron.b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()