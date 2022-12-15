from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
import scipy.io
from keras.utils import to_categorical
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from scipy.io import arff
import pandas as pd

data = arff.loadarff("diabetes.arff")
df = pd.DataFrame(data[0])

df.head()
target = df.iloc[:,0:3]
targetx = pd.get_dummies(target).idxmax(1)

X = np.asarray(df.iloc[:,:-1],dtype = np.float64)
y1 = df['class']
#y1 = np.asarray(df['class'],dtype = np.float64)
y = []
for i in range(len(y1)):
    y.append(int(y1[i].decode()))
n_train = int(0.9 * X.shape[0])
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

clf = BaggingClassifier(n_estimators=10, random_state=0).fit(trainX, trainy)
print(clf.score(testX, testy)*100)
train_sizes, train_scores, test_scores = learning_curve(clf, trainX, trainy, 
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
pred = clf.predict(testX)
import sklearn.metrics  
mse = sklearn.metrics.mean_squared_error(testy, pred)  
