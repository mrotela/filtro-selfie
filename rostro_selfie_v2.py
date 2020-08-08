# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:48:54 2020

@author: marcos rotela
"""

from modelo import *
import cv2
import numpy as np
#import imageio

# cargar el modelo
my_model = cargar_modelo('my_model')

# cargar el clasificador Face cascade para detectar rostros (clasificador preentrenado)
face_cascade = cv2.CascadeClassifier('cascadas/haarcascade_frontalface_default.xml')

###############################################################################
# definir los limites para los distintos colores
# limites de color rojo
rojoBajo1 = np.array([0, 100, 20])
rojoAlto1 = np.array([10, 255, 255])
rojoBajo2 = np.array([175, 100, 20])
rojoAlto2 = np.array([180, 255, 255])

# definir los limites para considerar el color azul
blueLower = np.array([100, 60, 60], np.uint8)
blueUpper = np.array([140, 255, 255], np.uint8)

# limites de color amararillo
amarilloBajo = np.array([20, 100, 20])
amarilloAlto = np.array([32, 255, 255])

# limites de color verde
verdeBajo = np.array([36, 100, 20])
verdeAlto = np.array([70, 255, 255])
###############################################################################

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# definir los filtros
filters = ['imagenes/sunglasses.png', 'imagenes/sunglasses_4.png', 'imagenes/sunglasses_6.png', 'imagenes/anounymus.png', 'imagenes/gafas_nariz_payaso.png', 'imagenes/estrella.png']


filterIndex = 5 ## predeterminado primera imagen de la lista
tipoFiltro = 3 ## predeterminado filtro para lentes
quitarFiltro = False ## bandera para habilitar o quitar los filtros del rostro
activarKeyPoints = False

# cargar camara
camera = cv2.VideoCapture(0)

# loop principal
while True:
    # tomar la pantalla actual
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    frame2 = np.copy(frame)
    ## cambiar el color BGR a HSV: el HSV es mejor para la detección de variaciones e intensidades de los colores
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ## convertir el frame a gris (para luego usar para detectar los rostros a partir del clasificador)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # agregar el boton "siguiente filtro" a la ventana    
    frame = cv2.rectangle(frame, (500,10), (620,65), (235,50,50), -1)          
    cv2.putText(frame, "filtro Next", (512, 37), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA) 
   
    # agregar caja amarilla para quitar filtro
    frame = cv2.rectangle(frame, (500,250), (620,205), (0,255,255), -1)  
    cv2.putText(frame, "filtro OFF", (512, 230), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    
    # agregar caja verde para habilitar filtro
    frame = cv2.rectangle(frame, (500,350), (620,305), (0,255,0), -1)  
    cv2.putText(frame, "filtro ON", (512, 330), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    
    # agregar caja roja para activar Keypoints
    frame = cv2.rectangle(frame, (10,10), (120,65), (0,0,255), -1)          
    cv2.putText(frame, "Keypoints", (22, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA) 
    
    ###########################################################################
    ## establecer tipoFiltro de acuerdo al index
    # tipoFiltro = 1 --> para lentes o similares
    # tipoFiltro = 2 --> para toda la cara
    # tipoFiltro = 3 --> para los ojos en forma individual
    if filterIndex == 0 or filterIndex == 1 or filterIndex == 2:
        tipoFiltro = 1
    else:
        if filterIndex == 3 or filterIndex == 4:
            tipoFiltro = 2
        else:
            tipoFiltro = 3    
    ###########################################################################

    # detectar rostros
    faces = face_cascade.detectMultiScale(gray, 1.25, 6)  ## (imagen, escala, minNeighbors) --> no se que significa minNeighbors
    
    # mascara para el color azul definido por los limites de lower y upper
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    # eliminar ruido de la mascara
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)
    
    cnts, _ = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    
    # verificar si se encontraron los contornos
    if len(cnts) > 0:
    	# buscamos el contorno mas grande    	
        # asumimos que el contorno corresponde a nuestro objeto
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]       
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)       
        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 255, 0), 2)        
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        
        if 500 <= center[0] <= 620 and 1 <= center[1] <= 65: # siguiente filtro
            filterIndex += 1
            filterIndex %= 6
            continue
    
    #imgBlue = cv2.drawContours(frame, cnts, 0, (0,255,255), 3)
    
    ###########################################################################
    # determinar que pixeles están en el cuadro amarillo
    # cv2.inRange se usa para determinar una mascara para detectar el color esperado
    amarilloMask = cv2.inRange(hsv, amarilloBajo, amarilloAlto)
    # eliminar ruido de la mascara
    amarilloMask = cv2.erode(amarilloMask, kernel, iterations=2)
    amarilloMask = cv2.morphologyEx(amarilloMask, cv2.MORPH_OPEN, kernel)
    amarilloMask = cv2.dilate(amarilloMask, kernel, iterations=1)    
   
    # encontrar los contornos del objeto que tenga el color del filtro
    cnts2, _ = cv2.findContours(amarilloMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center2 = None   
            
    # verificar si se encontraron los contornos
    if len(cnts2) > 0:
    	# buscamos el contorno mas grande    	
        # asumimos que el contorno corresponde a nuestro objeto
        cnt2 = sorted(cnts2, key = cv2.contourArea, reverse = True)[0]
        # encontrar el radio de los contornos 
        ((x, y), radius) = cv2.minEnclosingCircle(cnt2)
        # encontrar los contornos, en este caso como un circulo
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # obtener moments para conseguir el centro del objeto (circulo)
        M2 = cv2.moments(cnt2)
        if M2['m00'] != 0:
            center2 = (int(M2['m10'] / M2['m00']), int(M2['m01'] / M2['m00']))
        
            # se establecen los límites de los pixeles de la caja amarilla
            # en caso que el xy del objeto amarrillo entre dentro de caja, se realizan las acciones
            if 500 <= center2[0] <= 570 and 200 <= center2[1] <= 300:               
                quitarFiltro = True
                continue
    # print(quitarFiltro)        
    
    # dibujar el contorno del objeto identificado        
    #img = cv2.drawContours(frame, cnts2, 0, (0,255,255), 3)
    
    ###########################################################################
    # determinar que pixeles están en el cuadro verde
    verdeMask = cv2.inRange(hsv, verdeBajo, verdeAlto)     
    ## quitar ruido del verde
    verdeMask = cv2.erode(verdeMask, kernel, iterations=2)
    verdeMask = cv2.morphologyEx(verdeMask, cv2.MORPH_OPEN, kernel)
    verdeMask = cv2.dilate(verdeMask, kernel, iterations=1) 
    # encontrar el contorno de la mascara verde
    cntsVerde, _ = cv2.findContours(verdeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centerVerde = None   
           
    # verificar si se encontraron los contornos
    if len(cntsVerde) > 0:
    	# buscamos el contorno mas grande    	
        # asumimos que el contorno corresponde a nuestro objeto
        cntVerde = sorted(cntsVerde, key = cv2.contourArea, reverse = True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cntVerde)
        # Draw the circle around the contour
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        # Get the moments to calculate the center of the contour (in this case Circle)
        MV = cv2.moments(cntVerde)
        if MV['m00'] != 0:
            centerVerde = (int(MV['m10'] / MV['m00']), int(MV['m01'] / MV['m00']))
            #print(centerVerde)
            #if centerVerde[1] <= 65:
            if 500 <= centerVerde[0] <= 620 and 300 <= centerVerde[1] <= 350 and quitarFiltro is True:                    
                    filterIndex = 0                   
                    quitarFiltro = False
                    continue
            
    # dibujar el contorno del objeto identificado        
    #img = cv2.drawContours(frame, cntsVerde, 0, (0,255,0), 3)

    #print(quitarFiltro)   
    
    
    ###########################################################################
    # determinar que pixeles están en el cuadro rojo
    # cv2.inRange se usa para determinar una mascara para detectar el color esperado
    rojoMask = cv2.inRange(hsv, rojoBajo1, rojoAlto1)
    # eliminar ruido de la mascara
    rojoMask = cv2.erode(rojoMask, kernel, iterations=2)
    rojoMask = cv2.morphologyEx(rojoMask, cv2.MORPH_OPEN, kernel)
    rojoMask = cv2.dilate(rojoMask, kernel, iterations=1)    
   
    # encontrar los contornos del objeto que tenga el color del filtro
    cntsRojo, _ = cv2.findContours(rojoMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centerRojo = None   
            
    # verificar si se encontraron los contornos
    if len(cntsRojo) > 0:
    	# buscamos el contorno mas grande    	
        # asumimos que el contorno corresponde a nuestro objeto
        cntRojo = sorted(cntsRojo, key = cv2.contourArea, reverse = True)[0]
        # encontrar el radio de los contornos 
        ((x, y), radius) = cv2.minEnclosingCircle(cntRojo)
        # encontrar los contornos, en este caso como un circulo
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # obtener moments para conseguir el centro del objeto (circulo)
        M3 = cv2.moments(cntRojo)
        if M3['m00'] != 0:
            centerRojo = (int(M3['m10'] / M3['m00']), int(M3['m01'] / M3['m00']))
        
            # se establecen los límites de los pixeles de la caja rojo
            # en caso que el xy del objeto rojo entre dentro de caja, se realizan las acciones
            if 10 <= centerRojo[0] <= 120 and 30 <= centerRojo[1] <= 100:               
                activarKeyPoints = True
                continue
            
    # dibujar el contorno del objeto identificado        
    #imgRojo = cv2.drawContours(frame, cntsRojo, 0, (0,0,255), 3)
    
    #print(cntsRojo)
    #print(activarKeyPoints)
    # print(quitarFiltro)        
    
    
    ###########################################################################
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
       
        # predecir los Keypoints usando el modelo entrenado
        keypoints = my_model.predict(face_resized)

        # De-Normalize los valores de los keypoints 
        keypoints = keypoints * 48 + 48
       
        # mapear los keypoints con la imagen original 
        face_resized_color = cv2.resize(color_face, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized_color_sin_filtro = np.copy(face_resized_color)
        face_resized_color3 = np.copy(face_resized_color)          
       

        # cargar los Keypoints en un array
        points = []
        for i, co in enumerate(keypoints[0][0::2]):
            points.append((co, keypoints[0][1::2][i]))        
        
        if tipoFiltro == 1:
            sunglasses = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)
            sunglass_width = int((points[7][0]-points[9][0])*1.1)
            sunglass_height = int((points[10][1]-points[8][1])/1.1)
            sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation = cv2.INTER_CUBIC)
            transparent_region = sunglass_resized[:,:,:3] != 0
            face_resized_color[int(points[9][1]):int(points[9][1])+sunglass_height, int(points[9][0]):int(points[9][0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]
        else:
            if tipoFiltro == 2:
                # agregar el filtro (lentes o similares)
#                sunglasses = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)
#                sunglass_width = int((points[7][0]-points[9][0])*1.1)
#                sunglass_height = int((points[14][1]-points[8][1])/1.1)       
#                sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation = cv2.INTER_CUBIC)
#                transparent_region = sunglass_resized[:,:,:3] != 0
#                face_resized_color[int(points[9][1]):int(points[9][1])+sunglass_height, int(points[9][0]):int(points[9][0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]
#                
                sunglasses = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)
                sunglass_width = int((points[6][0]-points[9][0])*1.6)
                sunglass_height = int((points[14][1]-points[9][1])*1.1)     
                sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation = cv2.INTER_CUBIC)
                transparent_region = sunglass_resized[:,:,:3] != 0
                face_resized_color[int(points[9][1]):int(points[9][1])+sunglass_height, int(points[9][0]):int(points[9][0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]
            else:
                ##########################################################################################################
                #### aplicar filtro solo en los ojos                 
                filtro_ojo = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)
                filtro_ojo_ancho = int((points[3][0]-points[2][0])*2.1)
                filtro_ojo_alto = int((points[2][1]-points[6][1])*2.1)          
                filtro_ojo_resized = cv2.resize(filtro_ojo, (filtro_ojo_ancho, filtro_ojo_alto), interpolation = cv2.INTER_CUBIC)
                transparent_region = filtro_ojo_resized[:,:,:3] != 0
                # superponer imagen original con el filtro en el ojo derecho
                # el point 6 es el inicio de la ceja derecha
                face_resized_color3[int(points[6][1]):int(points[6][1])+filtro_ojo_alto, int(points[6][0]):int(points[6][0])+filtro_ojo_ancho,:][transparent_region] = filtro_ojo_resized[:,:,:3][transparent_region]
                frame[y:y+h, x:x+w] = cv2.resize(face_resized_color3, original_shape, interpolation = cv2.INTER_CUBIC)
                # superponer imagen original con el filtro en el ojo derecho + el filtro en el ojo izquierdo
                # el point 9 es el inicio de la ceja izquierda
                face_resized_color3[int(points[9][1]):int(points[9][1])+filtro_ojo_alto, int(points[9][0]):int(points[9][0])+filtro_ojo_ancho,:][transparent_region] = filtro_ojo_resized[:,:,:3][transparent_region]
               
                ##########################################################################################################  


            
        # en caso que el filtro este activado, vemos si el filtro es por ojo o por cara
        if quitarFiltro is False:
            if filterIndex == 5:
                 frame[y:y+h, x:x+w] = cv2.resize(face_resized_color3, original_shape, interpolation = cv2.INTER_CUBIC)
            else:
                frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)
        # si quitar filtro está activo entonces establecemos nuestro frame sin filtro
        else:
            frame[y:y+h, x:x+w] = cv2.resize(face_resized_color_sin_filtro, original_shape, interpolation = cv2.INTER_CUBIC)
            
        # agregar los keypoints al frame2
        for keypoint in points:
            cv2.circle(face_resized_color_sin_filtro, keypoint, 1, (0,0,255), 1)     
            #font = cv2.FONT_HERSHEY_PLAIN
            #cv2.putText(face_resized_color_sin_filtro, str(int(keypoint[0])), (int(keypoint[0]),int(keypoint[1])), font, 2,(0,0,0),1)            
            pointOjoDerecho = points[0] #el keypoint 0 corresponde al ojo derecho
            pointOjoIzquierdo = points[1] #el keypoint 1 corresponde al ojo izquierdo
            pointNariz = points[14] # el keypoint 10 corresponde a la nariz
        
        #cv2.circle(face_resized_color_sin_filtro, points[9], 1, (0,0,255), 3) 
        #cv2.circle(face_resized_color_sin_filtro, points[9], 1, (0,0,255), 3) 
        #cv2.circle(face_resized_color_sin_filtro, points[14], 1, (0,0,255), 5) 
        #cv2.circle(face_resized_color_sin_filtro, points[8], 1, (0,0,255), 5) 
        #if pointNariz:
        #    cv2.circle(face_resized_color2, pointNariz, 1, (0,0,255), 3) 
        #if pointOjoDerecho:
        #    cv2.circle(face_resized_color2, pointOjoDerecho, 1, (0,0,255), 5) 
        
        # obtenemos el frame sin filtro + Keypoints del rostro
        frame2[y:y+h, x:x+w] = cv2.resize(face_resized_color_sin_filtro, original_shape, interpolation = cv2.INTER_CUBIC)

        # mostrar los frames
        cv2.imshow("Filtro de selfie", frame)
        #cv2.imshow("Keypoints del rostro", frame2)

    # presionar la letra q para salir del loop
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break

# limpiar camara y cerrar las ventanas
camera.release()
cv2.destroyAllWindows()

