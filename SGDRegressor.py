
# ======================== Example 2: SGDRegressor on California Housing dataset ========================
# 1. What is SGD Regressor? (Definitions)
# SGD Regressor is a linear regression model trained using Gradient Descent
# It updates parameters one sample at a time
# Suitable for large-scale datasets
# Supports regularization (L1, L2, ElasticNet)

# 2. What does SGD Regressor do?
# Fits a linear model
# Minimizes loss function using gradient descent
# Updates weights incrementally
# Learns slope and intercept

# working
# Initialize weights randomly
# Pick one training sample
# Compute prediction error
# Update weights using gradient
# Repeat for many iterations


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data=fetch_california_housing()
x=data.data[:,[0]]
y=data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

sgd=SGDRegressor(max_iter=1000,tol=1e-3,eta0=0.01,learning_rate="constant",random_state=42)
sgd.fit(x_train,y_train)
y_pred=sgd.predict(x_test)
print("\ny_pred")
print(y_pred)

print("\nslope:",sgd.coef_[0])
print("intercept:",sgd.intercept_[0])
print("mean square error:",mean_squared_error(y_test,y_pred))
print("r2_score:",r2_score(y_test,y_pred))

plt.scatter(x_test,y_test,color="blue",alpha=0.5,label="data")
plt.plot(x_test,y_pred,color="red",label="regression line")
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("SGD Regression on California Housing Dataset")
plt.legend()
plt.legend()
plt.show()

