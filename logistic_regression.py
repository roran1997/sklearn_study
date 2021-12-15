import sys
sys.path.append('D:\\Code\\python\\machine-learning-toy-code\\ml-with-sklearn') # add current terminal path to sys.path
import numpy as np
from Mnist.load_data import load_local_mnist

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

(X_train, y_train), (X_test, y_test) = load_local_mnist(normalize = False,one_hot = False)

X_train, y_train= X_train[:2000], y_train[:2000]
X_test, y_test = X_test[:200],y_test[:200]

# solver：即使用的优化器，lbfgs：拟牛顿法， sag：随机梯度下降
model = LogisticRegression(solver='lbfgs', max_iter=1000) # lbfgs：拟牛顿法
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred)) # 打印报告