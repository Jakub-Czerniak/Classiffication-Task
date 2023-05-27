import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

data = pd.read_csv('Data/house-votes-84.data', sep=",", header=None, index_col=False)
data.columns=["democrat, republican",
   "handicapped-infants(y,n)",
   "water-project-cost-sharing(y,n)",
   "adoption-of-the-budget-resolution(y,n)",
   "physician-fee-freeze(y,n)",
   "el-salvador-aid(y,n)",
   "religious-groups-in-schools(y,n)",
   "anti-satellite-test-ban(y,n)",
   "aid-to-nicaraguan-contras(y,n)",
  "mx-missile(y,n)",
  "immigration(y,n)",
  "synfuels-corporation-cutback(y,n)",
  "education-spending(y,n)",
  "superfund-right-to-sue(y,n)",
  "crime(y,n)",
  "duty-free-exports(y,n)",
  "export-administration-act-south-africa(y,n)"]

data["democrat, republican"].hist(bins=3)
#plt.show()

X = data.drop("democrat, republican", axis=1)
y = data["democrat, republican"]

for column in X:
    X[column] = X[column].map({'y':1, '?':0 , 'n':-1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=3)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=3)

mlp_clf = MLPClassifier(solver='adam', random_state=3, max_iter=1000)
mlp_clf.fit(X_train,y_train)
mlp_pred = mlp_clf.predict(X_val)

tree_clf = DecisionTreeClassifier(random_state=3)
tree_clf.fit(X_train,y_train)
tree_pred = tree_clf.predict(X_val)

lsvc_clf = LinearSVC(random_state=3)
lsvc_clf.fit(X_train, y_train)
lsvc_pred = lsvc_clf.predict(X_val)

print("MultiLayerPerceptron")
print(accuracy_score(y_val, mlp_pred))
print(confusion_matrix(y_val, mlp_pred))

print("DecisionTree")
print(accuracy_score(y_val, tree_pred))
print(confusion_matrix(y_val, tree_pred))

print("LinearSupportVector")
print(accuracy_score(y_val, lsvc_pred))
print(confusion_matrix(y_val, lsvc_pred))