
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Book1.csv')
X = dataset.iloc[:, [0,1]].values
y = dataset.iloc[:, [2]].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

import keras
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(10, activation='relu',input_shape = (2,)))
classifier.add(Dense(10, activation='relu'))
classifier.add(Dense(10, activation='relu'))
classifier.add(Dense(1))

classifier.compile(optimizer='adam',loss='mse', metrics=['accuracy'])
classifier.fit(X_train,y_train,batch_size=256,epochs=1000)

ypred=classifier.predict(X_test)


