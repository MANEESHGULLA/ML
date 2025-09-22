# ======================== Example 1: SGDRegressor on synthetic linear data ========================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

np.random.seed(42)
x=2*np.random.rand(100,1)
y=3*x.flatten()+2+np.random.randn(100)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

sgd=SGDRegressor(max_iter=1000,tol=1e-3,eta0=0.1,learning_rate="constant",random_state=42)
sgd.fit(x_train,y_train)
y_pred=sgd.predict(x_test)
print("\ny_pred")
print(y_pred)

print("\nslope:",sgd.coef_[0])
print("intercept:",sgd.intercept_[0])
print("mean square error:",mean_squared_error(y_test,y_pred))
print("r2_score:",r2_score(y_test,y_pred))

plt.scatter(x,y,color="blue",label="data")
plt.plot(x_test,y_pred,color="red",label="regression line")
plt.legend()
plt.show()

# ======================== Example 2: SGDRegressor on California Housing dataset ========================


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

# ======================== Example 3: Multiple Linear Regression on custom dataset ========================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import pandas as pd

data = {
    "Size_sqft": [1000, 1500, 2000, 2500, 3000, 1800],
    "No_of_rooms": [2, 3, 3, 4, 5, 3],
    "Age_years": [10, 5, 20, 15, 8, 12],
    "Price": [200000, 250000, 270000, 350000, 400000, 280000]
}

df=pd.DataFrame(data)
x=df[['Size_sqft','No_of_rooms','Age_years']]
y=df[['Price']]

model=LinearRegression()
model.fit(x,y)
y_pred=model.predict(x)
print("\ny_pred")
print(y_pred)

print("\nslope:",model.coef_)
print("intercept:",model.intercept_)
print(pd.DataFrame({'actual price':y.values.ravel(),'predicted price':y_pred.ravel()}))

plt.scatter(y,y_pred,color="blue")
plt.plot([y.min(),y.max()],[y.min(),y.max()],color='red')
plt.xlabel("actual prce")
plt.ylabel("predicted Price")
plt.show()




