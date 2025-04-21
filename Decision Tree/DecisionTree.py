import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix

iris = load_iris()
x= pd.DataFrame(iris.data,columns=iris.feature_names)
y= pd.Series(iris.target)

x.describe()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

model = DecisionTreeClassifier(criterion='gini',random_state=0)
model.fit(x_train,y_train)

plt.figure(figsize=(12,12))
plot_tree(model,filled=True)
plt.show()

y_pred = model.predict(x_test)
print("Accuracy - ",accuracy_score(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,cmap='Blues',xticklabels=iris.target_names,yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

sample = [[1.2,2.5,0.2,1.5]]
prediction = model.predict(x_test)
print("Prediction Class - ",iris.target_names[prediction[0]])
