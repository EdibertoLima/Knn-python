# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import statistics 

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)

dataset.head()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

mean_acc = []



for i in range(1, 30):
    #alterar para numeros em vez de %
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30)
    #print(X_train)
    
    #normalizaçao
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    
    print(confusion_matrix(y_test, y_pred))
    acc=accuracy_score(y_test, y_pred)
    print(acc)
    mean_acc.append(acc)

print("A accuracia media é :" , statistics.mean(mean_acc))