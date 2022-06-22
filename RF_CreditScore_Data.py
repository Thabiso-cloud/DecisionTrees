import pandas as np
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

# Logistic Regression accuracy: 93%
# KNN: 97.5%
# KNN without normalizing: 84%

credit_data = pd.read_csv("D:\\Python Machine Learning Course\\PythonMachineLearning\\Datasets\\Datasets"
                          "\\credit_data.csv")

features = credit_data[['income', 'age', 'loan']]
targets = credit_data.default

# machine learning handles arrays not data-frames
X = np.array(features).reshape(-1,3)
y = np.array(targets)

model = RandomForestClassifier()

predicted = cross_validate(model, X, y, cv = 10)
print(np.mean(predicted['test_score']))