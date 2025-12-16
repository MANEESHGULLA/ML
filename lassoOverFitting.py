# What is Lasso Regression?
# Lasso Regression (Least Absolute Shrinkage and Selection Operator) is a regularized linear regression technique that:
# 👉 Reduces overfitting
# 👉 Shrinks coefficients
# 👉 Automatically performs feature selection

# lasso Regression Works 
# Starts with a normal linear regression model
# Adds an L1 penalty (absolute value of coefficients) to the loss function
# Penalizes large coefficient values
# Shrinks coefficients toward zero
# Forces some coefficients to become exactly zero
# Automatically removes irrelevant features
# Reduces overfitting
# Produces a simpler, sparse model


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. Load diabetes dataset
X, y = load_diabetes(return_X_y=True)


# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#3. Training on linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_train_pred_lr = lin_reg.predict(X_train)
y_test_pred_lr = lin_reg.predict(X_test)

print("Linear Regression:")
print("Train R²:", r2_score(y_train, y_train_pred_lr))
print("Test R²:", r2_score(y_test, y_test_pred_lr))

# 3. Define different alpha values (regularization strengths)
alphas = np.logspace(-3, 2, 50)  # Lambda values in-between 0.001 → 100
train_scores = []
test_scores = []

# Lasso Regression
lasso = Lasso(alpha=0.6)   # regularization strength
lasso.fit(X_train, y_train)

y_train_pred_lasso = lasso.predict(X_train)
y_test_pred_lasso = lasso.predict(X_test)

print("\nLasso Regression:")
print("Train R²:", r2_score(y_train, y_train_pred_lasso))
print("Test R²:", r2_score(y_test, y_test_pred_lasso))


# 4. Train Lasso model for each alpha and record scores
for a in alphas:
    lasso = Lasso(alpha=a, max_iter=10000)
    lasso.fit(X_train, y_train)

    y_train_pred_lasso = lasso.predict(X_train)
    y_test_pred_lasso = lasso.predict(X_test)


    train_scores.append(r2_score(y_train, y_train_pred_lasso))
    test_scores.append(r2_score(y_test, y_test_pred_lasso))

# 5. Plot Train vs Test R² scores
plt.figure(figsize=(8,6))
plt.semilogx(alphas, train_scores, label="Train R²", marker='o')
plt.semilogx(alphas, test_scores, label="Test R²", marker='s')
plt.xlabel("Alpha (λ)")
plt.ylabel("R² Score")
plt.title("Lasso Regression: Reducing Overfitting on Diabetes Dataset")
plt.legend()
plt.grid(True)
plt.show()
