# What is Correlation

# Correlation measures how strongly two variables are related

# It shows how one feature changes when another feature changes

# Correlation value lies between -1 and +1

# Definitions of Correlation

# Positive correlation:
# Both variables increase or decrease together
# Example: income ↑ → house price ↑

# Negative correlation:
# One variable increases while the other decreases
# Example: distance from city ↑ → house price ↓

# Zero correlation:
# No relationship between variables

# How Correlation Works (Points)

# Take two features

# Compare how their values change together

# Calculate correlation value

# Result shows:

# Direction (positive / negative)

# Strength (weak / strong)

from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns

data=fetch_california_housing()
x=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

plt.figure(figsize=(12,10))
cor=x_train.corr()
sns.heatmap(cor,annot=True,cmap=plt.cm.CMRmap_r)
plt.title("Feature Correlation Heatmap")
plt.show()


def correlation(dataset,threshold):
  col_corr=set()
  corr_matrix=dataset.corr()

  for i in range(len(corr_matrix.columns)):
    for j in range(i):
      if abs(corr_matrix.iloc[i,j])>threshold:
        col_name=corr_matrix.columns[i]
        col_corr.add(col_name)

  return col_corr

corr_features= correlation(x_train,0.8)
corr_features

def correlation_pairs(dataset,threshold):
  corr_pairs=[]
  corr_matrix=dataset.corr()
  for i in range(len(corr_matrix.columns)):
    for j in range(i):
      if abs(corr_matrix.iloc[i,j])>threshold:
        col_i=corr_matrix.columns[i]
        col_j=corr_matrix.columns[j]
        corr_pairs.append((col_i,col_j,corr_matrix.iloc[i,j]))
  return corr_pairs

co_pairs=correlation_pairs(x_train,0.8)
co_pairs

x_train.drop(corr_features,axis=1)

x_test.drop(corr_features,axis=1)


