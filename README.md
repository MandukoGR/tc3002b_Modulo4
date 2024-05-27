# Clasificación de imágenes de expresiones faciales con CNN
# Conjunto de datos inicial
El conjunto de datos utilizado para este proyecto se obtuvo de [Kaggle](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)

Este conjunto de datos contiene 35,887 imágenes en escala de grises de rostros humanos con diferentes expresiones faciales. Las imágenes tienen un tamaño de 48x48 píxeles. 


Este conjunto de datos ya se encuentra dividido en dos subconjuntos: train y test, con una proporción de 80% para entrenamiento y 20% para evaluación. Cada subconjunto se compone de siete carpetas, cada una de las cuales representa una clase de expresión facial:


| Clase          | Test (cantidad de imágenes) | Train (cantidad de imágenes) |
| -------------- | --------------------------- | ---------------------------- |
| angry          | 958                         | 3995                         |
| disgusted      | 111                         | 436                          |
| fearful        | 1024                        | 4097                         |
| happy          | 1774                        | 7215                         |
| neutral        | 1233                        | 4965                         |
| sad            | 1247                        | 4830                         |
| surprised      | 831                         | 3171                         |


# División de conjunto de datos
Se decidió modificar la estructura del conjunto de datos. Esto para generar un mejor balance de datos y también incluir un subconjunto de datos destinado a validación.

En primer lugar, se redujo el número de clases de 7 a 5 eliminando las clases de "disgusted" y "surprised" debido a su poca cantidad de datos en comparación con las otras clases.

Posteriormente, el conjunto de datos fue dividido en tres subconjuntos:
- **Train**: Es el subconjunto de datos que se utilizará para entrenar al modelo
- **Test**: Los datos en este subconjunto se utilizarán para evaluar el rendimiento del modelo con imágenes nuevas que no han sido utilizadas para el entrenamiento del modelo.
- **Validation**: Los datos en este subconjunto serán utilizados para evaluar el rendimiento del modelo durante el entrenamiento y poder ajustar los hiperparámetros del modelo antes de evaluar el modelo con el conjunto de prueba.

Se realizó una división del 70%, 15% y 15% de la totalidad de los datos, siendo el primer conjunto para entrenamiento, el segundo para validación y el tercero para pruebas. Para esto, se renombraron las imagenes de train y test de cada clase utiizando el programa "rename.py", debido a que contaban con el mismo nombre y no era posible combinar las carpetas. Después de renombrar las imágenes se combinaron las carpetas train y test y se utilizó el programa "split_data.py" obtenido de [Kaggle](https://www.kaggle.com/code/shuvostp/split-folders-for-train-test-val-split-of-images) para hacer la división deseada.

Por lo tanto, la estructura final del conjunto de datos cuenta con un total de 31,339 imágenes, donde estas se distribuyen de la siguiente manera:

| Clase          | Test (cantidad de imágenes) | Train (cantidad de imágenes) | Validation (cantidad de imágenes) |
| -------------- | --------------------------- | ---------------------------- | --------------------------------- |
| angry          | 744                         | 3467                         | 742
| fearful        | 769                         | 3584                         | 768
| happy          | 1349                        | 6292                         | 1349
| neutral        | 931                         | 4338                         | 929
| sad            | 913                         | 4253                         | 911

Es importante mencionar que la correcta división de los datos evita tener problemas como sobreentrenamiento o subentrenamiento. El sobreentrenamiento u **overfitting** ocurre cuando el modelo se ajusta perfectamente o casi perfectamente a los datos de entrenamiento pero tiene un mal rendimiento con datos nuevos. Esto puede ocurrir si se utiliza la totalidad de un conjunto de datos para entrenar un modelo. Por otro lado, el subentrenamiento o **underfitting** sucede cuando la arquitectura del modelo no es lo suficientemente compleja o cuando los datos de entrenamiento no son suficientes.


# Preprocesado de datos y técnicas de escalamiento

Para realizar el preprocesado de las imágenes, en este proyecto se utiliza el generador de datos **ImageDataGenerator** de Tensorflow. Este cuenta con parámetros configurables que permiten indicar al generador las modificaciones que se realizarán en las imágenes. En este caso, se modificaran los siguientes parámetros:

- **Rotación**: La imágenes pueden ser rotadas hasta 180 grados, ya que esto no altera las características de una expresión facial.
- **Brillo**: Se puede ajustar el brillo de las imagenes mientras la imagen siga siendo visible, esto puede realizarse utilizando un rango entre 0.1 y 0.9 en la propiedad brightness_range de ImageDataGenerator.
- **Reescalamiento**: Escalamiento de los valores de los píxeles de las imágenes dividiendo el valor de cada píxel entre 255.
- **Zoom**: Se utilizarán valores entre menores a 1 para solamente generar acercamientos. Es importante que el acercamiento no sea demasiado porque se pueden perder partes importantes de la imagen.
- **Flip horizontal**: El flip horizontal refleja la imagen de forma horizontal. En este conjunto de datos se puede utilizar debido a que no altera las expresiones faciales.

Otros parámetros de ImageDataGenerator como shear o width shifting no fueron utilizados debido a que pueden alterar la expresión del un rostro.

Finalmente los parámetros utilizados fueron:
```python
train_datagen = ImageDataGenerator(
							rescale = 1./255, # Reescalamiento
							rotation_range = 180, # Rotación de 180 grados
                            brightness_range= (0.1, 0.9), # Nivel de brillo
							zoom_range = 0.2, # Zoom
							horizontal_flip = True,) # Reflejo horizontal
```
Se realizó el código "preprocess.py" para probar el ImageDataGenerator configurado. Se utilizó un batch de 10 para que se generen 10 imágenes que son almacenadas en la carpeta augmented que se encuentra dentro de data.


# Modelo
La arquitectura del modelo es la propuesta por Bodapati y Strilakshmi en el artículo de investigación "A Deep CNN Architecture for Facial Expression Recognition in the Wild.". Esta consiste de 5 bloques convulucionales, donde cada bloque cuenta con dos capas convulucionales con filtros de 4x4 inicializadas con "Xavier weight initialization" , una capa de normalización de batches, un maxpooling de 2x2 y un dropout de 0.5. Finalmente se incluye una capa de aplanamiento y una capa densa con un tamaño de 5 neuronas (en la arquitectura original son 7 pero debido a la reducción de clases que se realizó ahora es 5). En la siguiente imagen se ilustra la arquitectura elegida. 
![Arquitectura](/images/architecture.png)

El framework elegido para realizar el modelo fue keras de tensorflow. El modelo es el siguiente:

```python

initializer = initializers.GlorotNormal()

def build_model(input_shape=(48, 48, 1), num_classes=5):
    model = tf.keras.models.Sequential([
        # First convolution block
        tf.keras.layers.Conv2D(32, (4, 4), activation='relu', kernel_initializer=initializer, input_shape=input_shape, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (4, 4), activation='relu',kernel_initializer=initializer, padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),
        
        # Second convolution block (similar to others)
        tf.keras.layers.Conv2D(64, (4, 4), activation='relu', kernel_initializer=initializer, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (4, 4), activation='relu', kernel_initializer=initializer, padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),
        
        # Third convolution block
        tf.keras.layers.Conv2D(128, (4, 4), activation='relu',kernel_initializer=initializer, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (4, 4), activation='relu',kernel_initializer=initializer, padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),
        
        # Fourth convolution block
        tf.keras.layers.Conv2D(128, (4, 4), activation='relu',kernel_initializer=initializer, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (4, 4), activation='relu',kernel_initializer=initializer, padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),
        
        # Fifth convolution block
        tf.keras.layers.Conv2D(256, (4, 4), activation='relu',kernel_initializer=initializer, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (4, 4), activation='relu',kernel_initializer=initializer, padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),
        
        # Flatten the output for dense layer
        tf.keras.layers.Flatten(),
        
        # Dense layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

```

Para compilar y entrenar el modelo se utilizaron 50 epochs con un tamaño de batch de 64. Los hiperparámetros utilizados, también obtenidos del artículo de investigación y son los siguientes:

| Hiperparámetro | Valor                      |
| -------------- | -------------------------- |
| Loss           | Categorical Crossentropy   |
| Optimizer      | Adam                       | 
| Learning rate  | 0.000001                   |
| Weight decay   | 0.001                      |
| momentum       | 0.9                        |


```python

# Set hyperparameters
epochs = 50
learning_rate = 0.001
loss = 'categorical_crossentropy'
batch_size = 64
momentum = 0.9
weight_decay = 0.000001

# Define optimizer
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, beta_1=momentum, decay=weight_decay)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1= momentum)

# Build the model
model = build_model()
model.compile(loss=loss,optimizer=optimizer, metrics=['acc'])

history = model.fit(train_generator, validation_data = val_generator, epochs = epochs, batch_size = batch_size)


```

# Diagnóstico de resultados

## Resultados
|                 | Train           | Validation       | Test          |
| --------------  | --------------- | ---------------- | ------------- |
| accuracy        | 0.6782          | 0.6592           | 0.6387        |
| loss            | 0.8462          | 0.9161           | 0.9334        |


Estos resultados indican que hay overffiting. Ya que como se muestra en la tabla, la diferencia de loss entre train y test es de aproximadamente 8%. De acuerdo al siguiente gráfico, podemos inferir que hay overfitting.

![Gráfico de referencia](/images/ref_graf.jpeg)

En las siguientes imagenes se puede observar el comportamiento de accuracy y loss durante los epoch.

![Gráfico acc y loss](/images/acc_loss_grafs.png)

Al obtener la matriz de confusión y las métricas de precision, recall y F1 score, se puede notar que no hay buenos resultados.

![Matriz](/images/matrix_report.png)

Hay un gran número de predicciones para cada clase que no fueron las correctas. Esto se puede deber a una o varias de las siguientes opciones:

1. Desbalance de Clases:
La clase happy tiene un mayor número de imágenes que las otras clases (con una diferencia entre 2000 y 3000 aproximadamente), es decir, el modelo tiene muchas más muestras de una clase que de las otras, por lo que el modelo podría estar aprendiendo a predecir siempre la clase mayoritaria. Esto puede resultar en una alta cantidad de verdaderos positivos (TP) pero también en una alta cantidad de falsos positivos (FP) y falsos negativos (FN), lo cual afectará negativamente las métricas de precisión, recall y F1 score.

1. Sobreajuste (Overfitting):
El modelo puede estar sobreajustando a los datos de entrenamiento, especialmente con una arquitectura tan compleja y profunda. Esto significa que el modelo podría estar aprendiendo patrones específicos del conjunto de entrenamiento que no generalizan bien a los datos de validación o prueba.

1. Errores en el Preprocesamiento de Datos:
Los datos deben estar correctamente preprocesados para asegurar que el modelo pueda aprender de ellos de manera efectiva. Agregar parámetros en el ImagedataGenerator podría ayudar a generalizar más los datos y que el modelo tenga un mejor comportamiento con datos que nunca ha utilizado.







# Referencias
1. P. Verma, V. Aggrawal, and J. Maggu, "FExR.A-DCNN: Facial Emotion Recognition with Attention Mechanism using Deep Convolution Neural Network," in *Proc. 2022 Fourteenth Int. Conf. Contemporary Computing (IC3-2022)*, New York, NY, USA, 2022, pp. 196–203. doi: 10.1145/3549206.3549243.
2. Bodapati, J.D., Srilakshmi, U. & Veeranjaneyulu, N. FERNet: A Deep CNN Architecture for Facial Expression Recognition in the Wild. J. Inst. Eng. India Ser. B 103, 439–448 (2022). https://doi.org/10.1007/s40031-021-00681-8
3. A. Ananthu, "Emotion Detection FER Dataset," Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer?select=train. [Accessed: 19-May-2024].
4. S. P. Shuvo, "split_folders for train test val split of images," Kaggle, 2023. [Online]. Available: https://www.kaggle.com/code/shuvostp/split-folders-for-train-test-val-split-of-images. [Accessed: 19-May-2024].


