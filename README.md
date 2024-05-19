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


# Referencias
1. P. Verma, V. Aggrawal, and J. Maggu, "FExR.A-DCNN: Facial Emotion Recognition with Attention Mechanism using Deep Convolution Neural Network," in *Proc. 2022 Fourteenth Int. Conf. Contemporary Computing (IC3-2022)*, New York, NY, USA, 2022, pp. 196–203. doi: 10.1145/3549206.3549243.
2. A. Ananthu, "Emotion Detection FER Dataset," Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer?select=train. [Accessed: 19-May-2024].
3. S. P. Shuvo, "split_folders for train test val split of images," Kaggle, 2023. [Online]. Available: https://www.kaggle.com/code/shuvostp/split-folders-for-train-test-val-split-of-images. [Accessed: 19-May-2024].


