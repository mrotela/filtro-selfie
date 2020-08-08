# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:48:54 2020

@author: marcos rotela
"""

from modelo import *
import cv2
import numpy as np

# cargar el modelo
my_model = cargar_modelo('my_model')

# Face cascade para detectar rostros
face_cascade = cv2.CascadeClassifier('cascadas/haarcascade_frontalface_default.xml')

# definir los limites para considerar el color azul
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# definir los limites para considerar el color amarillo
yellowLower = np.array([207, 255, 249])
yellowUpper = np.array([0, 220, 255])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# definir los filtros
filters = ['imagenes/sunglasses.png', 'imagenes/sunglasses_2.png', 'imagenes/sunglasses_3.jpg', 'imagenes/sunglasses_4.png', 'imagenes/sunglasses_5.jpg', 'imagenes/sunglasses_6.png', 'imagenes/anounymus.jpg']
filterIndex = 0

# cargar camara
camera = cv2.VideoCapture(0)

# loop
while True:
    # tomar la pantalla actual
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    frame2 = np.copy(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # agregar el boton "siguiente filtro" a la ventana    
    frame = cv2.rectangle(frame, (500,10), (620,65), (235,50,50), -1)          
    cv2.putText(frame, "sgte filtro", (512, 37), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    # agregar otro rectangulo para otros filtros
    frame = cv2.rectangle(frame, (500,120), (620,65), (255,255,0), -1)  
    cv2.putText(frame, "otro filtro", (512, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # detectar rostros
    faces = face_cascade.detectMultiScale(gray, 1.25, 6)

    # determinar que pixeles están en el cuadro azul
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)
    print(blueMask)
    # determinar que pixeles están en el cuadro amarillo
    yellowMask = cv2.inRange(hsv, yellowLower, yellowUpper)
    yellowMask = cv2.erode(yellowMask, kernel, iterations=2)
    yellowMask = cv2.morphologyEx(yellowMask, cv2.MORPH_OPEN, kernel)
    yellowMask = cv2.dilate(yellowMask, kernel, iterations=1)

    # encontrar contornos en la imagen
    #(_, cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
    cnts, _ = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
    center = None
    
    cnts2, _ = cv2.findContours(yellowMask.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
    center2 = None
    
    # verificar si se encontraron los contornos
    if len(cnts) > 0:
    	# buscamos el contorno mas grande    	
        # asumimos que el contorno corresponde a nuestro objeto
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # Draw the circle around the contour
        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 255, 0), 2)
        # Get the moments to calculate the center of the contour (in this case Circle)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if center[1] <= 65:
            if 500 <= center[0] <= 620: # Next Filter
                filterIndex += 1
                filterIndex %= 7
                continue
            
    # verificar si se encontraron los contornos
    if len(cnts2) > 0:
    	# buscamos el contorno mas grande    	
        # asumimos que el contorno corresponde a nuestro objeto
        cnt2 = sorted(cnts2, key = cv2.contourArea, reverse = True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt2)
        # Draw the circle around the contour
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Get the moments to calculate the center of the contour (in this case Circle)
        M2 = cv2.moments(cnt2)
        center2 = (int(M2['m10'] / M2['m00']), int(M2['m01'] / M2['m00']))
        
        if center2[1] <= 65:
            if 500 <= center2[0] <= 620: # Next Filter
                #filterIndex += 1
                #filterIndex %= 7
                continue

    for (x, y, w, h) in faces:

        # obtener cara
        gray_face = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]

        # Normalizar para que coincida con el formato de entrada del modelo - Rango de pixel [0, 1]
        gray_normalized = gray_face / 255

        # cambiar tamaño 96x96 para que coincida con el formato de entrada del modelo
        original_shape = gray_face.shape # copia para futuras referencias
        face_resized = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized_copy = face_resized.copy()
        face_resized = face_resized.reshape(1, 96, 96, 1)
       
        # predecir los keypoints usando el modelo
        keypoints = my_model.predict(face_resized)

        # De-Normalize los valores de los keypoints 
        keypoints = keypoints * 48 + 48
       
        # mapear los keypoints con la imagen original 
        face_resized_color = cv2.resize(color_face, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized_color2 = np.copy(face_resized_color)

        # Pair them together
        points = []
        for i, co in enumerate(keypoints[0][0::2]):
            points.append((co, keypoints[0][1::2][i]))

        # agregar el filtro
        sunglasses = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)
        sunglass_width = int((points[7][0]-points[9][0])*1.1)
        sunglass_height = int((points[10][1]-points[8][1])/1.1)
        sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation = cv2.INTER_CUBIC)
        transparent_region = sunglass_resized[:,:,:3] != 0
        face_resized_color[int(points[9][1]):int(points[9][1])+sunglass_height, int(points[9][0]):int(points[9][0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]

        # cambiar el tamaño face_resized_color de vuelta a su forma original
        frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)
     
        # agregar los keypoints al frame2
        for keypoint in points:
            cv2.circle(face_resized_color2, keypoint, 1, (0,255,0), 1)

        frame2[y:y+h, x:x+w] = cv2.resize(face_resized_color2, original_shape, interpolation = cv2.INTER_CUBIC)

        # mostrar los frames
        cv2.imshow("Filtro de selfie", frame)
        cv2.imshow("Keypoints del rostro", frame2)

    # presionar la letra q para salir del loop
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break

# limpiar camara y cerrar las ventanas
camera.release()
cv2.destroyAllWindows()
