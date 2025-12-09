#ridge
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

x,y=load_diabetes(return_X_y=True)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

lr=LinearRegression()
lr.fit(x_train,y_train)

y_train_pred_lr=lr.predict(x_train)
y_test_pred_lr=lr.predict(x_test)
print("linear")
print("Train r2 Linear:",r2_score(y_train,y_train_pred_lr))
print("Test r2 Linear:",r2_score(y_test,y_test_pred_lr))
print()
ridge=Ridge(alpha=10)
ridge.fit(x_train,y_train)

y_train_pred_ridge=ridge.predict(x_train)
y_test_pred_ridge=ridge.predict(x_test)

print("ridge")
print("Train r2 ridge:",r2_score(y_train,y_train_pred_ridge))
print("Test r2 ridge:",r2_score(y_test,y_test_pred_ridge)) 

alphas=np.logspace(-3,3,50)
train_scores=[]
test_scores=[]

for a in alphas:
  ridge=Ridge(alpha=a,max_iter=10000)
  ridge.fit(x_train,y_train)

  y_train_pred_ridge=ridge.predict(x_train)
  y_test_pred_ridge=ridge.predict(x_test)


  train_scores.append(r2_score(y_train,y_train_pred_ridge))
  test_scores.append(r2_score(y_test,y_test_pred_ridge))

plt.figure(figsize=(8,6))
plt.semilogx(alphas,train_scores,label="train_r2")
plt.semilogx(alphas,test_scores,label="test_r2")
plt.xlabel("alpha")
plt.ylabel("r2 scores")
plt.legend()
plt.grid()
plt.show()
