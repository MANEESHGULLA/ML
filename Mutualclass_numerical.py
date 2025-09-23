from sklearn.datasets import load_iris
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
X_df = pd.DataFrame(X, columns=feature_names)

# Select top 2 features using Mutual Information
selector = SelectKBest(mutual_info_classif, k=2)
X_reduced = selector.fit_transform(X, y)

# Get selected features directly
selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
print("Selected features:", selected_features)

# Keep only selected features
X_df = X_df[selected_features]
print("\nFinal Extracted DataFrame:\n", X_df.head())
