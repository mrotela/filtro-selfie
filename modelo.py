# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:42:40 2020

@author: marcos rotela

descargada de https://towardsdatascience.com/facial-keypoints-detection-deep-learning-737547f73515
"""
# cargamos las clases para generar modelos secuenciales
from keras.models import Sequential
from keras.models import load_model
# cargamos las capas
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

def obtener_modelo():
    '''
    La red debe aceptar una imagen en escala de grises de 96x96 como entrada, y debe generar un vector con 30 entradas,
    correspondiente a las ubicaciones predichas (horizontal y vertical) de 15 puntos clave faciales.
    '''
    # modelo secuencia de layers 
    model = Sequential()
    
    #bloque1
    model.add(Convolution2D(32, (5, 5), input_shape=(96,96,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #bloque2
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    #bloque3
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    #bloque4
    model.add(Convolution2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # se agrega layer para aplanar valores de entrada
    model.add(Flatten())

    # bloque fully-connectec
    # dense: tipo de layer para agregar neuronas
    # relu: rectificador lineal, funcion que ejecuta cada neurona
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(30))

    return model;

def compilar_modelo(model, optimizer, loss, metrics):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def entrenar_modelo(model, X_train, y_train):
    return model.fit(X_train, y_train, epochs=100, batch_size=200, verbose=1, validation_split=0.2)

def guardar_modelo(model, fileName):
    model.save(fileName + '.h5')

def cargar_modelo(fileName):
    return load_model(fileName + '.h5')

