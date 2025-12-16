# POLYNOMIAL REGRESSION

# 1. What is Polynomial Regression? (Definitions)
# Polynomial Regression is an extension of Linear Regression
# It models non-linear relationships between input x and output y
# Uses powers of input features to fit curves

# 2. What does Polynomial Regression find?
# 1. Finds best-fitting curve
# 2. Uses linear regression on transformed polynomial features
# 3. Minimizes sum of squared errors
# 4. Predicts non-linear patterns

# 3. Polynomial Regression Equation
# y = b0 + b1x + b2x^2 + ... + bnx^n

# n → degree of polynomial
# b0, b1, ... → coefficients
# x^i → polynomial term


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

x=6* np.random.rand(200,1)-3
y=0.8*x**2+0.9*x+2+np.random.randn(200,1)

plt.plot(x,y,"b.")
plt.show()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("r2 score linear:",r2_score(y_test,y_pred))

plt.plot(x,y,"b.")
plt.plot(x_train,lr.predict(x_train),"r-")
plt.show()

poly=PolynomialFeatures(degree=2,include_bias=True)
x_train_poly=poly.fit_transform(x_train)
x_test_poly=poly.transform(x_test)

lr_poly=LinearRegression()
lr_poly.fit(x_train_poly,y_train)
y_poly_pred=lr_poly.predict(x_test_poly)
print("r2 score polynomial:",r2_score(y_test,y_poly_pred))


x_new=np.linspace(-3,3,200).reshape(200,1)
x_new_poly=poly.transform(x_new)
y_new=lr_poly.predict(x_new_poly)

plt.plot(x_train,y_train,"b.",label="training points")
plt.plot(x_test,y_test,"g.",label="testing points")
plt.plot(x_new,y_new,"r-",linewidth=2,label="prediction")
plt.legend()
plt.show()
