from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np

X = np.array([
    [0, 2, 0, 3],
    [0, 1, 4, 3],
    [0, 1, 1, 3],
    [0, 1, 0, 3],
    [0, 1, 3, 3]
])
columns = ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D']
df = pd.DataFrame(X, columns=columns)

sel = VarianceThreshold(threshold=0)
df_selected = pd.DataFrame(sel.fit_transform(df), columns=df.columns[sel.get_support()])
print(df_selected)






import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder

# Sample categorical dataset
data = {
    'Color': ['Red', 'Red', 'Blue', 'Blue', 'Green'],
    'Size': ['S', 'S', 'M', 'L', 'L'],
    'Shape': ['Circle', 'Circle', 'Circle', 'Circle', 'Circle']
}

# Create DataFrame
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, drop=None)
X_encoded = encoder.fit_transform(df)
encoded_feature_names = encoder.get_feature_names_out(df.columns)
df_encoded = pd.DataFrame(X_encoded, columns=encoded_feature_names)
print("\nOne-Hot Encoded DataFrame:\n", df_encoded)

# Apply Variance Threshold
sel = VarianceThreshold(threshold=0.0)
X_reduced = sel.fit_transform(df_encoded)
selected_columns = df_encoded.columns[sel.get_support()] 
df_reduced = pd.DataFrame(X_reduced, columns=selected_columns)
print("\nReduced DataFrame (after VarianceThreshold):\n", df_reduced)
