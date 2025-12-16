import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = {
    "Size_sqft": [1000, 1500, 2000, 2500, 3000, 1800],
    "No_of_rooms": [2, 3, 3, 4, 5, 3],
    "Age_years": [10, 5, 20, 15, 8, 12],
    "Price": [200000, 250000, 270000, 350000, 400000, 280000]
}
df = pd.DataFrame(data)
print(df)
X = df[["Size_sqft", "No_of_rooms", "Age_years"]]
y = df["Price"]
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
print("Intercept (b):", model.intercept_)
print("Coefficients (w1, w2, w3):", model.coef_)
print(pd.DataFrame({"Actual Price": y, "Predicted Price": y_pred}))
plt.scatter(y, y_pred, color="blue")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
