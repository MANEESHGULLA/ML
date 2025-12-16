# MUTUAL INFORMATION (NUMERICAL FEATURES)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# 1. Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
feature_names = X.columns

# 2. Select top-k features using Mutual Information
k = 2
mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)
X_reduced = mi_selector.fit_transform(X_df, y)

# 3. Get MI scores
mi_scores = mi_selector.scores_
mi_feature_scores = pd.Series(mi_scores,index=feature_names).sort_values(ascending=False)

print("🔹 Mutual Information Scores:")
print(mi_feature_scores)

# 4. Get selected feature names
mi_selected_features = feature_names[mi_selector.get_support()].tolist()
print("\n🔹 Selected features by MI:", mi_selected_features)

# 5. Build reduced DataFrame
X_reduced = pd.DataFrame(X_reduced, columns=mi_selected_features)
print("\n🔹 Reduced Dataset (MI – Top", k, "features):")
print(X_reduced.head())
