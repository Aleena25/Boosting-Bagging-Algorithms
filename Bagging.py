# bagging mlp ensemble on blobs dataset
from sklearn.datasets import make_blobs
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
import numpy as np
from numpy import array
from numpy import argmax
from scipy.io import arff
import pandas as pd

data = arff.loadarff("diabetes.arff")
df = pd.DataFrame(data[0])

df.head()

X = np.asarray(df.iloc[:,:-1],dtype = np.float64)
y1 = df['class']
#y1 = np.asarray(df['class'],dtype = np.float64)
y = []
for i in range(len(y1)):
    y.append(int(y1[i].decode()))




from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.tree import DecisionTreeClassifier
bag_model = BaggingClassifier(
base_estimator=DecisionTreeClassifier(), 
n_estimators=100, 
max_samples=0.8, 
bootstrap=True,
oob_score=True,
random_state=0
)

    
bag_model.fit(X, y)
bag_model.oob_score_
train_sizes, train_scores, test_scores = learning_curve(bag_model, X, y, 
                                                        cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.subplots(1, figsize=(10,10))
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()
#MSE
pred = bag_model.predict(newX)

import sklearn.metrics  
mse = sklearn.metrics.mean_squared_error(newy, pred)  


