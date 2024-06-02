# Entregables finales 

### Proyectos realizados para la materia de inteligencia artificial. Enero - Junio 2024.

### Jonathan Giovani Ceja Contreras
### No. control: 20120091
---
# 1. Convolutional Neural Network (CNN) 
El proyecto consiste en usar redes neuronales convolucionales como modelos de clasificación de imágenes. La implementación clasificara imágenes con las siguientes 5 clases o etiquetas:
  
* Asaltos
* Inundación
* Incendios
* Robos a casa habitación
* Tornados

A continuación, se explicara a detalle el código requerido para la implementación.

### 1.1 Librerías o dependencias requeridas
Python 3.9

Las siguientes librerias de Python:
* **cv2**: pip install opencv-contrib-python
* **numpy 1.26.4**: pip install numpy
* **matplotlib 3.9.0**: pip install matplotlib
* **scikit-learn 1.5.0**: pip install scikit-learn
* **keras 3.3.3**: pip install keras
* **tensorflow 2.16.1**: pip install tensorflow

### 1.2 Función para capturar frames de videos
Para el proceso de la recolección del dataset de las imágenes para alimentar al modelo con las categorías para clasificar, se utiliza el siguiente código:

Primero se importa la librería **cv2** y **os** 

``` python

import cv2 as cv
import os
```
Después, se define en una constante el nombre del **directorio** en donde se deberá guardar los frames capturados, el video **cap** del que se tomaran los frames, luego se define una variable **i** como contador y un segundo **contador** que toma el total de archivos en el directorio de destino, esto para no sobreescribir las imágenes existentes. 

``` python
directorio = "robocasa"
cap = cv.VideoCapture("../assets/videos/robo casa/robocasa f.mp4")
i=0
contador=len(os.listdir(directorio))
```
Luego se define una ultima constante, para controlar el intervalo de frames que deben pasar para guardar el frame del video. Debido a que, si no se controla el intervalo de frames, se generará un dataset con demasiadas imágenes "repetidas", ya que no habría diferencia significativa entre cada imagen al ser capturada frame a frame.

```python
frame_interval = 10  # Capturar una imagen cada n frames
```

En seguida, se entra en un ciclo infinito que va leyendo del video hasta que finalice la duración del video o se presione la tecla ESC (27), en el cuerpo de la función se valida si el contador es múltiplo del intervalo de frames definidos, entonces muestra el frame en una ventana con resize, para finalmente guardar el frame en el directorio destino.
``` python
while True:
    ret, frame = cap.read()
    if not ret:
        break    
    if i % frame_interval == 0:
        resized_frame = cv.resize(frame, (28, 21))
        cv.imshow('img', resized_frame)
        cv.imwrite(f'{directorio}/data{contador}.jpg', resized_frame)
        contador+=1
        
    i=i+1    
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()

```

### 1.3 Función para eliminar archivos de un directorio
Otra función que fue requerida en el proceso de generar el dataset de imágenes, es la de eliminar todos archivos de un directorio especifico para descartar rápidamente las imágenes incorrectas y volver a repetir el proceso de captura.
```python
directorio = "robocasa"
eliminar_archivos_directorio(directorio)
```
El cuerpo de la función se muestra a continuación.
```python
import os
def eliminar_archivos_directorio(directorio):
    # Verifica si el directorio existe
    if not os.path.exists(directorio):
        print(f"El directorio {directorio} no existe.")
        return

    # Recorre todos los archivos en el directorio
    for archivo in os.listdir(directorio):
        ruta_archivo = os.path.join(directorio, archivo)
        
        # Verifica si es un archivo antes de eliminar
        if os.path.isfile(ruta_archivo):
            os.remove(ruta_archivo)
            print(f"Archivo eliminado: {ruta_archivo}")

```

### 1.4 Importación de librerías
Para la ejecución de los siguientes bloques de código, se requieren de las siguientes importaciones
```python
import numpy as np
import os
import re
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential,Model, load_model
from tensorflow.keras.layers import Input
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Conv2D
)
from keras.layers import LeakyReLU
from tensorflow.keras.optimizers import SGD
```

### 1.5 Carga del dataset de imágenes

Definición de variables. dirname guarda la ruta completa de donde se encuentra el directorio del dataset de imágenes. 
* **os.getcwd()** obtiene el directorio de trabajo actual.
* **os.path.join(os.getcwd(), 'dataset/')** combina el directorio de trabajo actual con 'dataset2/' para formar la ruta completa al directorio dataset2.
* **os.sep** es el separador de directorios específico del sistema operativo (por ejemplo, '/' en Unix o '' en Windows). Al final se guarda la ruta completa en **imgpath**
  
```python
dirname = os.path.join(os.getcwd(),'dataset/')
imgpath = dirname + os.sep 

images = []
directories = []
dircount = []
prevRoot=''
cant=0
```

Después, en el cuerpo de la función se hace un doble ciclo, en el primero se itera por cada directorio encontrado el la ruta especificada, y el segundo itera por cada imagen dentro del directorio.

Dentro del segundo ciclo, se hace una validación de tipo de archivo, para aceptar solo **jpg|jpeg|png|bmp|tiff** como formato de imagen valida. Por cada iteración, se guarda las imágenes leídas en el directorio, los directorios recorridos y la suma total de las imágenes en cada subdirectorio.

```python
for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            if(len(image.shape)==3):
                
                images.append(image)
            b = "Leyendo..." + str(cant)
            print (b, end="\r")
            if prevRoot !=root:
                print(root, cant)
                prevRoot=root
                directories.append(root)
                dircount.append(cant)   
                cant=0

dircount.append(cant)
dircount = dircount[1:]
dircount[0]=dircount[0]+1
print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))

```
### 1.6 Creación de etiquetas y categorías
Se definen el total de etiquetas en una variable llamada **labels**
```python
labels=[]
indice=0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice=indice+1
print("Cantidad etiquetas creadas: ",len(labels))
```
Después, se guarda en una lista, el nombre de los subdirectorios que representaran a las **categorías para clasificar**
```python
deportes=[]
indice=0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice , name[len(name)-1])
    deportes.append(name[len(name)-1])
    indice=indice+1
```
### 1.7 Creación de set de Entrenamientos y Test

Primero se convierten la lista de etiquetas a un numpy y la lista de imágenes

```python
y = np.array(labels)
X = np.array(images, dtype=np.uint8) #convierto de lista a numpy

# Find the unique numbers from the train labels
classes = np.unique(y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

```
Despues de utiliza la función de **train_test_split** de scikit-learn
* **X**: Las características del conjunto de datos, que serian las categorías o etiquetas.
* **y**: Los valores objetivo correspondientes a X, las cuales serian las imágenes.
* **test_size=0.2**: Especifica que el **20%** del conjunto de datos debe ser asignado al conjunto de prueba, mientras que el **80%** restante se asigna al conjunto de entrenamiento.

Como salidas obtenemos las siguientes:
* **train_X**: Las características del conjunto de entrenamiento.
* **test_X**: Las características del conjunto de prueba.
* **train_Y**: Las etiquetas del conjunto de entrenamiento.
* **test_Y**: Las etiquetas del conjunto de prueba.

```python
train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.2)
```
Finalmente, se muestra la primer y segunda imagen del conjunto de entrenamiento, con el siguiente código:

```python
plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))
```
![resultados de entrenamiento y test](markdown/assets/cnn/image.png)

### 1.8 Creación el One-hot Encoding para la red
Esta estrategia consiste en crear una columna binaria (que solo puede contener los valores 0 o 1) para cada valor único que exista en la variable categórica que estamos codificando, y marcar con un 1 la columna correspondiente al valor presente en cada registro, dejando las demás columnas con un valor de 0. 

```python
# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)
```
Utilizamos la función de **to_categoria** de keras.utils que convierte etiquetas categóricas a una matriz de codificación one-hot
Teniendo una salida como se muestra a continuación:
```python
# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

Original label: 1
After conversion to one-hot: [0. 1. 0. 0. 0.]
```

### 1.9 Creación del modelo CNN
A continuación, se definen algunas constantes necesarias que representan los parámetros para crear el modelo CNN.

```python
#declaramos variables con los parámetros de configuración de la red
INIT_LR = 1e-3 # Valor inicial de learning rate. El valor 1e-3 corresponde con 0.001
epochs = 20 # Cantidad de iteraciones completas al conjunto de imagenes de entrenamiento
batch_size = 64 # cantidad de imágenes que se toman a la vez en memoria
```

Después, se guarda en la variable sport_model la instancia **Sequencial**. La clase Sequential permite construir modelos de manera lineal, donde las capas se apilan una tras otra.

```python
sport_model = Sequential()
sport_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(21,28,3)))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(MaxPooling2D((2, 2),padding='same'))
sport_model.add(Dropout(0.5))


sport_model.add(Flatten())
sport_model.add(Dense(32, activation='linear'))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(Dropout(0.5))
sport_model.add(Dense(nClasses, activation='softmax'))
```
En el proceso de creación del modelo, se agregan capas como:
* **Conv2D**: Agrega una capa convolucional con 32 filtros y un tamaño de kernel de (3, 3), tambien se especifica el tamaño de las imágenes del conjunto, el cual fue de 21x28 para cada imagen.
* **LeakyReLU**: Agrega una capa LeakyReLU con un coeficiente alpha de 0.1 para evitar el problema de unidades muertas
* **MaxPooling2D**: Reduce las dimensiones espaciales de la salida mediante una ventana de 2x2. Con el padding='same': Mantiene las dimensiones de salida iguales a las de entrada.
* **Dropout(0.5)**: Apaga aleatoriamente el 50% de las neuronas durante el entrenamiento para prevenir el sobreajuste.
* **Flatten()**: Convierte la salida 2D en una única dimensión (vector) para la entrada en la capa densa.
* **Dense(32, activation='linear')**: Agrega una capa densa con 32 neuronas y activación lineal.

Se agrega la ultima capa de activación:
* **Dense(nClasses, activation='softmax')**: Agrega una capa densa con un número de neuronas igual al número de clases (nClasses). Con activation='softmax', utiliza la activación softmax para obtener probabilidades para cada clase.

Al ejecutar la función **summary()** podemos ver un resumen de las capas del modelo creado

![Resumen de las capas del modelo](markdown/assets/cnn/summary.png)

Por ultimo, compilamos el modelo, usando la siguiente instrucción.

```python
sport_model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(learning_rate=INIT_LR, decay=INIT_LR / 100),
    metrics=['accuracy']
)
```

### 1.10 Entrenamiento del modelo

Para entrenar el modelo, usamos la función **fit**, mandando los parámetros requeridos. Entre ellos, datos obtenidos en el entrenamiento y test realizado anteriormente, etiquetas y los datos de validación.

```python
sport_train = sport_model.fit(
    train_X, train_label,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(valid_X, valid_label)
)
```

Finalmente, se guarda el modelo entrenado para no volver a repetir el proceso cuando se quiera volver a usar.

```python 
sport_model.save("cnn.h5")
```

### 1.11 Cargar un modelo entrenado
Si se requiere cargar un modelo que ya este entrenado, se realizan las siguientes instrucciones:

```python
# Ruta del archivo .h5
model_path = 'cnn_temp.h5'

# Cargar el modelo
sport_model = load_model(model_path)

# Verifica la estructura del modelo
sport_model.summary()
```

### 1.12 Evaluar la red
Utilizando la función **evaluate()** podemos evaluar la red entrenada, se le manda los siguientes datos:

* **test_X**: Representa a los datos de entrada de prueba
* **test_Y_one_hot**: Las etiquetas de prueba correspondientes
* **verbose**: Un entero que controla la verbosidad del proceso. 0 significa silencio, 1 significa barra de progreso.

```python
test_eval = sport_model.evaluate(test_X, test_Y_one_hot, verbose=1)
```
Teniendo como resultado:
* Test loss: 0.20288848876953125
* Test accuracy: 0.9549939036369324

Además, es posible graficar la exactitud y de imprecisión, usando la librería matplotlib con el siguiente código:

```python
accuracy = sport_train.history['accuracy']
val_accuracy = sport_train.history['val_accuracy']
loss = sport_train.history['loss']
val_loss = sport_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
``` 

Teniendo un resultado como se muestra en las siguientes imágenes, para exactitud e imprecisión, respectivamente.

![precision red cnn](markdown/assets/cnn/precision_cnn.png)

![imprecision red cnn](markdown/assets/cnn/imprecision_cnn.png)

Después, podemos evaluar la cantidad de etiquetas correctas e incorrectas, con el siguiente código:

```python
predicted_classes2 = sport_model.predict(test_X)
predicted_classes=[]
for predicted_sport in predicted_classes2:
    predicted_classes.append(predicted_sport.tolist().index(max(predicted_sport)))
predicted_classes=np.array(predicted_classes)
```
Primero se obtiene los valores de predicción con la función **predict**, mandando el conjunto de datos de entrenamiento **test_X**

Después, se muestra la cantidad de **etiquetas correctas** y una muestra de 9 imágenes del conjunto de **entrenamiento**

Found 7066 correct labels

![resultados_correctos_cnn](markdown/assets/cnn/resultados_correctos_cnn.png)

Por otro lado, se muestra la cantidad de **etiquetas incorrectas** y otra muestra de 9 imágenes que muestran la imprecisión del modelo al confundir clases.

```python
incorrect = np.where(predicted_classes!=test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect].reshape(21,28,3), cmap='gray', interpolation='none')
    plt.title("{}, {}".format(deportes[predicted_classes[incorrect]],
                                                    deportes[test_Y[incorrect]]))
    plt.tight_layout()
```

Found 333 incorrect labels

![resultados_incorrectos_cnn](markdown/assets/cnn/resultados_incorrectos_cnn.png)

Finalmente, podemos obtener una tabla en donde se muestra la precisión de cada clase, asi como otros parametros.

```python
target_names = ["Class {}".format(i) for i in range(nClasses)]
```
0 asalto
1 incendios
2 inundacion
3 robocasa
4 tornados

              precision    recall  f1-score   support

     Class 0       0.95      0.96      0.95      1960
     Class 1       0.99      1.00      1.00      1298
     Class 2       0.90      0.90      0.90      1367
     Class 3       0.97      0.96      0.97      1549
     Class 4       0.96      0.96      0.96      1225

    accuracy                           0.95      7399
   macro avg       0.96      0.96      0.96      7399
weighted avg       0.96      0.95      0.95      7399

### 1.13 Función para probar el modelo

El siguiente código sirve para mostrar el resultado que devuelve el modelo con imágenes de entrada. En la lista **filenames** se escribe la ruta de la o las imágenes de entrada a clasificar, después, cada imagen de entrada se muestra en un frame con un texto que indica a la clase que pertenece usando la función **predict**.

```python
from skimage.transform import resize
import cv2 as cv

images=[]

#incendios
filenames = ['incendio.jpg','incendio.jpeg','incendio3.jpeg']
```

Se itera por cada imagen en la lista de imágenes, guardando la imagen redimensionada a 21x28 y la lista resultante se convierte a numpy en una variable llamada **test_X**

```python
for filepath in filenames:
    image = plt.imread(f'test/{filepath}',0)
    image_resized = resize(image, (21, 28),anti_aliasing=True,clip=False,preserve_range=True)
    images.append(image_resized)

X = np.array(images, dtype=np.uint8) #convierto de lista a numpy
test_X = X.astype('float32')
test_X = test_X / 255.

```

Luego se definen algunas constantes para la configuración del frame con la imagen de entrada y resultado

```python
# Especificar las nuevas dimensiones (ancho, alto)
nuevo_ancho = 600
nuevo_alto = 400
dimensiones = (nuevo_ancho, nuevo_alto)

posicion = (0, nuevo_alto)  # (x, y)

# Tamaño de la fuente
escala_fuente = 1

# Colores
color_texto = (255, 255, 255)  # Blanco
color_fondo = (0, 0, 0)        # Negro

# Grosor del texto
grosor_texto = 2

# Fuente del texto
fuente = cv.FONT_HERSHEY_DUPLEX

```

Finalmente, se usa la función **predict**. predicted_classes contiene las predicciones, donde cada predicción es un vector de probabilidades para cada clase. Convierte **img_tagged** a una lista (img_tagged.tolist()), encuentra el índice de la probabilidad máxima (img_tagged.tolist().index(max(img_tagged))) y utiliza este índice para encontrar la etiqueta correspondiente en la lista de clases.

```python
#Predecir clases
predicted_classes = sport_model.predict(test_X)

for i, img_tagged in enumerate(predicted_classes):
    print(filenames[i], deportes[img_tagged.tolist().index(max(img_tagged))])

    # # Leer la imagen
    imagen = cv.imread(f'test/{filenames[i]}')
    # Redimensionar la imagen
    imagen_redimensionada = cv.resize(imagen, dimensiones)

    # Especificar el texto y su posición
    texto = deportes[img_tagged.tolist().index(max(img_tagged))]

    # Obtener el tamaño del texto
    (tamaño_texto, _) = cv.getTextSize(texto, fuente, escala_fuente, grosor_texto)

    # Calcular las coordenadas del rectángulo de fondo
    posicion_inferior_izquierda = posicion
    posicion_superior_derecha = (posicion[0] + tamaño_texto[0], posicion[1] - tamaño_texto[1] - 10)


    # Dibujar el rectángulo de fondo
    cv.rectangle(imagen_redimensionada, posicion_inferior_izquierda, posicion_superior_derecha, color_fondo, cv.FILLED)

    # Dibujar el texto sobre el rectángulo
    cv.putText(imagen_redimensionada, texto, posicion, fuente, escala_fuente, color_texto, grosor_texto)

    # Mostrar la imagen con el texto
    cv.imshow('Imagen', imagen_redimensionada)

    # Esperar a que se presione una tecla para cerrar la ventana
    cv.waitKey(0)
    cv.destroyAllWindows()

```
A continuación, se muestran capturas de los resultados obtenidos.

Incendios

![alt text](markdown/assets/cnn/incendios2.png)

Inundación

![alt text](markdown/assets/cnn/inundacion.png)

Tornados

![alt text](markdown/assets/cnn/tornados.png)

Robo casa

![alt text](markdown/assets/cnn/robocasa.png)

Asalto

![alt text](markdown/assets/cnn/asalto.png)