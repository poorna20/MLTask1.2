
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Book1.csv')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

import keras
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(10, activation='relu',input_shape = (1,)))
classifier.add(Dense(10, activation='relu'))
classifier.add(Dense(1))

classifier.compile(optimizer='adam',loss='mean_absolute_error', metrics=['accuracy'])
classifier.fit(X_train,y_train,batch_size=256,nb_epoch=1000)

ypred=classifier.predict(X_test)

plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,(classifier.predict(X_train)) , color = 'blue')
plt.title('2*x(Training set)')
plt.xlabel('X')
plt.ylabel('Result')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test,(ypred) , color = 'blue')
plt.title(' 2*X (Test set)')
plt.xlabel('X')
plt.ylabel('Result ')
plt.show()

