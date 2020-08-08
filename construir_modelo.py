# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:28:28 2020

@author: marcos rotela
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

from modelo import *

test = False
FTRAIN = 'datos/training.csv'
FTEST = 'datos/test.csv'
fname = FTEST if test else FTRAIN
df = read_csv(os.path.expanduser(fname))  # load dataframes

# The Image column has pixel values separated by space; convert
# the values to numpy arrays:
df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

df = df.dropna()  # drop all rows that have missing values in them

X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1] (Normalizing)
X = X.astype(np.float32)
X = X.reshape(-1, 96, 96, 1) # return each images as 96 x 96 x 1

if not test:  # only FTRAIN has target columns
    y = df[df.columns[:-1]].values
    y = (y - 48) / 48  # scale target coordinates to [-1, 1] (Normalizing)
    X, y = shuffle(X, y, random_state=42)  # shuffle train data
    y = y.astype(np.float32)
else:
    y = None

#return X, y

# cargar training set
X_train = X
y_train = y

# obtener el modelo
my_model = obtener_modelo()

# compilar la red neuronal convolucional configurado el optimizador, loss y metrica
compilar_modelo(my_model, optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# entrenar el modelo
hist = entrenar_modelo(my_model, X_train, y_train)

# train_my_CNN_model returns a History object. History.history attribute is a record of training loss values and metrics
# values at successive epochs, as well as validation loss values and validation metrics values (if applicable).

# guardar el modelo
guardar_modelo(my_model, 'my_model')

'''
# You can skip all the steps above (from 'obtener el modelo') after running the script for the first time.
# Just load the recent model using load_my_CNN_model and use it to predict keypoints on any face data
my_model = load_my_CNN_model('my_model')
'''