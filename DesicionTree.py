from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree,DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score

x,y=load_iris(return_X_y=True)

feature_names=load_iris().feature_names
class_names=load_iris().target_names

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

dtc=DecisionTreeClassifier(criterion="entropy",max_depth=5,random_state=42)
dtc.fit(x_train,y_train)

plt.figure(figsize=(12,8))
plot_tree(
    dtc,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()

print("Class of the Flower:",dtc.predict([[3,15,4,1.5]]))
print("Class of the Flower:",dtc.predict([[5.1,3.5,1.4,0.2]]))
print("Class of the Flower:",dtc.predict([[5.9,3.,5.1,1.8]]))
pred=dtc.predict([[3,1.5,4,2]])[0]
print("class of the flower:",pred,"-",class_names[pred])

y_pred=dtc.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy of Decision Tree:", acc)
print("confusion matrix:\n",confusion_matrix(y_test,y_pred))
