# CHI-SQUARE FEATURE SELECTION (CATEGORICAL TARGET + NON-NEGATIVE NUMERICAL FEATURES)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

# 1. Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
feature_names = X.columns

# 2. Scale features to non-negative values (required for chi2)
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

# 3. Select top-k features using Chi-Square

chi2_selector = SelectKBest(score_func=chi2, k=2)
X_reduced = chi2_selector.fit_transform(X_scaled, y)

# 4. Chi-Square scores
chi2_scores = chi2_selector.scores_
chi2_feature_scores = pd.DataFrame({
    'Feature': feature_names,
    'Chi2 Score': chi2_scores
}).sort_values(by='Chi2 Score', ascending=False)

print("🔸 Chi-Square Scores:")
print(chi2_feature_scores)

# 5. Selected features
chi2_selected_features = feature_names[chi2_selector.get_support()].tolist()
print("\n🔸 Selected features by Chi-Square:", chi2_selected_features)

# 6. Reduced dataset
X_reduced_chi2_df = pd.DataFrame(X_reduced, columns=chi2_selected_features)
print("\n🔸 Reduced Dataset (Chi-Square – Top", 2, "features):")
print(X_reduced_chi2_df.head())
