#ordinary least sqaures

#linear regression
#Linear Regression models the relationship between independent variables (X) and a dependent variable (y) using a straight line.
#Simple Linear Regression
#y=mx+c

# 1. What is Ordinary Least Squares? (Definitions)
# OLS is a method to estimate parameters of Linear Regression
# It finds the best-fit straight line
# It minimizes the sum of squared errors
# Used for continuous value prediction

# 2. What does OLS find?
# Finds the best slope (m)
# Finds the best intercept (b)
# Minimizes prediction error
# Models linear relationship

# 3. Linear Regression Equation
# y=mx+b

# Where:
# m → slope
# b → intercept
# y → predicted value
# x → input feature

# 4. What is “Least Squares”?
# Error = actual − predicted
# Square the errors
# Add all squared errors
# Choose  𝑚,b
# m,b that minimize this sum

# 5. Does OLS work like this?
# Assume a straight line
# Compute prediction errors
# Square the errors
# Minimize total squared error
# Get optimal slope & intercept

# 6. Goal of OLS (One line)
# ➡️ Find the best-fit line by minimizing squared prediction errors


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error ,r2_score
import numpy as np
df=pd.read_csv("placement.csv")

plt.scatter(df['cgpa'],df['package'])
plt.xlabel("cgpa")
plt.ylabel("package")
plt.show()

x=df.iloc[:,0:1]
y=df.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

lr=LinearRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)
print("\ny_pred")
print(y_pred)

print("\nexample prediction")
print("cgpa:",x.iloc[0].values)
print("predicted package:",lr.predict(x.iloc[0].values.reshape(1,1)))

plt.scatter(df['cgpa'],df['package'])
plt.plot(x_train,lr.predict(x_train),color="red")
plt.show()

m=lr.coef_
b=lr.intercept_

print("\nslope:",m)
print("\nintercept:",b)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print(f'\nMean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'R-squared: {r2:.2f}')

print("package (cgpa=8.58):",m*8.58+b)
print("package (cgpa=8.58):",lr.predict([[8.58]]))
