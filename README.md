# filtro-selfie
Proyecto de filtro para selfies (usando Open CV + Keras + Python)

Módulo de Redes Neurales - Diplomado Machine Learning

Lo que hace esta aplicación es identificar rostros (usando un clasificador preentrenado) desde la cámara web del computador (emula el selfie), predecir los puntos claves del rostro (a partir de un modelo de red neuronal entrenado) y aplicar filtros sobre el rostro. Para esto, se utilizan algunas funciones que utilizan detección de objetos con  colores y pixeles dentro de la pantalla para ir ejecutando cada función emulando botones en la pantalla.

Pre-requisitos

Se deben instalar en el ambiente Open CV, Keras (Tensor Flow), Python (versión 3.7 o superior)

Para el desarrollo del proyecto se realizaron los siguientes pasos:

1- Entrenar modelo para detectar los keypoints de una cara

1.1- Los datos se descargan de Kaggle https://www.kaggle.com/c/facial-keypoints-detection/data

1.2- los datos de training.csv es el dataset utilizado para el entrenamiento del modelo

1.3- definición del modelo de red neuronal (modelo.py) -- se usa modelo de https://towardsdatascience.com/facial-keypoints-detection-deep-learning-737547f73515

1.4- build del modelo y se guarda el modelo generado

2- Descargar el clasificador preentrenado para deteccion de rostros

2.1- se utiliza el clasificador de cv2 haarcascade_frontalface_default.xml

3- Gestionar filtros en rostro 

3.1- detectar los rostros utilizado el clasificador 

3.2- una vez detectado el rostro, se ejecuta el predictor de keypoints

3.3- a partir de los keypoints se establecen referencias para los distintos tipos de filtros implementados

3.3.1- se definieron 3 tipos de filtros: por cada ojo, para ambos ojos y para el rostro

3.3.2- funciones a partir de la detección de objetos en cuadros por cada color

caja azul: siguiente filtro
caja verde: activar filtro
caja amarilla: desactivar filtro
