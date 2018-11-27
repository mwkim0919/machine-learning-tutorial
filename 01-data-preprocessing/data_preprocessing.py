#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 19:16:14 2018

@author: minwoo
"""

# Data preprocessing

# Importing libraries
import numpy
import matplotlib.pyplot as pyplot
import pandas

# Importing dataset
dataset = pandas.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
# missing data would be mean of existing data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# Problem: This would mean that France (e.g. 1) is greater than Germany(e.g. 0) or the othey way around
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
"""
Train set gets used for machine learning and then using test set, 
we will determine if machine can correctly predict the outcome of test set.
"""
# 20% of the data would be test set (20% is in the convention)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""
This is to prevent one variable dominating the other.
In this case, salary will dominate age.
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)