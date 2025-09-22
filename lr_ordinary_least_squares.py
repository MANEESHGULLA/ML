#ordinary least sqaures

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
