import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

# 1. Load dataset (Iris for demo)
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
# Converts target array into a Pandas Series for easier handling.
# Target classes: 0 → setosa; 1 → versicolor; 2 → virginica
print(X)

# 2. Scale features to non-negative values (required by chi2)
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(X_scaled)
# MinMaxScaler scales each feature to the range [0, 1].
# fit() computes min and max; transform() scales the values.
# We wrap the output as a DataFrame and retain original column names.

# 3. Apply Chi-Square test
k = 2  # Number of top features to select
chi2_selector = SelectKBest(score_func=chi2, k=k)
X_reduced = chi2_selector.fit_transform(X_scaled, y)
# fit(): Computes Chi-Square scores between each feature and the target.
# transform(): Selects top k features based on scores.
# X_reduced: NumPy array with only top k features.

# 4. Get Chi-Square scores
scores = chi2_selector.scores_
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': scores
}).sort_values(by='Chi2 Score', ascending=False)
# Chi-Square scores are sorted in descending order.

# 5. Display results
print("Chi-Square Scores for Each Feature:")
print(feature_scores)

print("\nReduced Dataset (Top", k, "features):")
selected_columns = X.columns[chi2_selector.get_support()]
# get_support() returns a boolean array indicating selected features.[False,False,True,True]
# This masks X.columns to return selected feature names.

X_reduced_df = pd.DataFrame(X_reduced, columns=selected_columns)
print(X_reduced_df.head())
