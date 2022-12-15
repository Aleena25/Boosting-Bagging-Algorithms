from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd

data = arff.loadarff(r"D:\36uci\diabetes.arff")
df = pd.DataFrame(data[0])

df.head()


X = np.asarray(df.iloc[:,:-1],dtype = np.float64)
y1 = df['class']
#y1 = np.asarray(df['class'],dtype = np.float64)
y = []
for i in range(len(y1)):
    y.append(int(y1[i].decode()))
# one hot encode output variable

# split into train and test
n_train = int(0.9 * X.shape[0])
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

# define model
num_hidden = int(X.shape[1]/2)
if type(trainy)==list:
    num_out = 1
    trainy = np.asarray(trainy,dtype = np.float64)
    testy = np.asarray(testy,dtype = np.float64)
else:

    num_out = int(y.shape[1])
input_dim = X.shape[1]
model = Sequential()
model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))# here 4 is half of the featres. no: of features is 9
model.add(Dense(num_out, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=50, verbose=0)
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# learning curves of model accuracy
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

#MSE
pred = model.predict(testX)
import sklearn.metrics  
mse = sklearn.metrics.mean_squared_error(testy, pred)  
