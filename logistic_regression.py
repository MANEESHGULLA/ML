import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sn
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]

df = pd.read_csv(url, names=columns)

print("First 5 rows of the dataset:")
print(df.head())
print(df.shape)
print()

x=df.drop("class",axis=1)
y=df['class']

scaler=StandardScaler()
x_sclaed=scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

lr=LogisticRegression(max_iter=1000)
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

print("\nconfusion matrix")
cm=confusion_matrix(y_test,y_pred)
print(cm)

print("\naccuracy score")
print(accuracy_score(y_test,y_pred))

print("\nnumber of non diabetic persons")
num_non_diabetic=(y_test==0).sum()
print(num_non_diabetic)

print("\nnumber of diabetic persons")
num_diabetic=(y_test==1).sum()
print(num_diabetic)

plt.figure(figsize=(8,6))
sn.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
