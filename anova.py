from sklearn.datasets import load_iris
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler  

# Load dataset
data = load_iris()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
feature_names = x.columns

# Scale features (recommended for ANOVA)
scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=feature_names)

# Apply ANOVA F-test
selector = SelectKBest(f_classif, k=2)
x_reduced = selector.fit_transform(x_scaled, y)

# ANOVA F-scores
anova_scores = selector.scores_
anova_feature_scores = pd.DataFrame({
    'feature': feature_names,
    'anova_scores': anova_scores
}).sort_values(by='anova_scores', ascending=False)

print(anova_feature_scores)

# Selected features
selected_features = feature_names[selector.get_support()].tolist()
print(selected_features)

# Reduced dataset
x_reduced_df = pd.DataFrame(x_reduced, columns=selected_features)
print(x_reduced_df.head())
