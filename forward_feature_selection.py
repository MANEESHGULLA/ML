import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load Iris dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
print(X)
#y = pd.Series(data.target)
y = data.target
#print(y)
#Create a Pandas Series from the target values of the dataset and store it in the variable y


# 2. Split into training and testing sets (optional but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Standardize features (good for models like logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(X_train_scaled)
X_test_scaled = scaler.transform(X_test)
#fit(X_train) → Calculates the scaling parameters
 #(e.g., mean and standard deviation for StandardScaler) from the training data.
 #transform the data by applying scaling



# 4. Define model
model = LogisticRegression(max_iter=200)

# 5. Apply Forward Feature Selection
selector = SequentialFeatureSelector(
    estimator=model,
    n_features_to_select=2,        # choose top 2 features
    direction='forward',           # forward selection
    scoring='accuracy',            # or 'f1_macro', 'roc_auc_ovr', etc.
    cv=5                           # 5-fold cross-validation
)

selector.fit(X_train_scaled, y_train)

# 6. Get selected features
selected_features = X.columns[selector.get_support()]
print("Selected Features:")
print(selected_features)

# 7. Reduced dataset
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
print("\nReduced Training Set:")
print(X_train_selected.head())
