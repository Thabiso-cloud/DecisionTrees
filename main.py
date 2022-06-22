import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets
from sklearn.model_selection import GridSearchCV

iris_data = datasets.load_iris()

features = iris_data.data
targets = iris_data.target

# with grid search you can find an optimal parameter "parameter tuning"
param_grid = {'max_depth': np.arange(1, 10)}

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)




model = DecisionTreeClassifier(criterion='gini')

predicted = cross_validate(model, features, targets, cv = 10)
print(np.mean(predicted['test_score']))

# in every iteration we split the data randomly in cross validation + DecisionTreeClassifier
# initializes the tree randomly: that's why we get different results

tree = GridSearchCV(DecisionTreeClassifier(), param_grid)
tree.fit(feature_train, target_train)
print("Best parameter with Grid search: ", tree.best_params_)

grid_predictions = tree.predict(feature_test)
print(confusion_matrix(target_test, grid_predictions))
print(accuracy_score(target_test, grid_predictions))

