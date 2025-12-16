# What is Sequential Feature Selection (SFS)
# Sequential Feature Selection is a wrapper-based feature selection method
# It selects features based on model performance
# Uses a machine learning model to evaluate feature combinations
# Forward Feature Selection
# Starts with zero features
# Adds one feature at a time
# At each step:
# Tries all remaining features
# Picks the one that improves model accuracy the most
# Stops when required number of features is selected


# How Forward Selection Works (Points)
# Start with empty feature set
# Train model with each single feature
# Select feature with highest accuracy
# Add selected feature to feature set
# Repeat until desired number of features is reached


# What is the Goal of This Code 
# 👉 To select the best 2 features that give highest classification accuracy using Logistic Regression


from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load Iris dataset
data = load_iris()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
feature_names = x.columns

# 2. Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# 3. Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 4. Define model
model = LogisticRegression(max_iter=200)

# 5. Forward Feature Selection
selector = SequentialFeatureSelector(
    estimator=model,
    n_features_to_select=2,
    direction="forward",
    scoring="accuracy",
    cv=5
)
selector.fit(x_train_scaled, y_train)

# 6. Selected features
selected_features = feature_names[selector.get_support()].tolist()
print("Selected features:", selected_features)

# 7. Reduced datasets (unscaled)
x_train_reduced = x_train[selected_features]
x_test_reduced = x_test[selected_features]

print("\nReduced Training Set:")
print(x_train_reduced.head())
