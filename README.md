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


La arquitectura del modelo es la propuesta por Bodapati y Strilakshmi en el artículo de investigación "A Deep CNN Architecture for Facial Expression Recognition in the Wild.". La cual obtuvo un 69% de test accuracy sin utilizar registro de puntos de referencia faciales antes de realizar la clasificación. Además la arquitectura es simple comparada con arquitecturas como la propuesta por Amil Khanzada, Charles Bai y Ferhat Turker Celepcikay en el artículo "Facial Expression Recognition with Deep Learning" que alcanza hasta un 73.2% de test accuracy utilizando transfer learning pero con un tiempo de entrenamiento de por lo menos 20 horas (con el equipo de cómputo que tengo disponible).
La arquitectura consiste de 5 bloques convulucionales, donde cada bloque cuenta con dos capas convulucionales con filtros de 4x4 inicializadas con "Xavier weight initialization" , una capa de normalización de batches, un maxpooling de 2x2 y un dropout de 0.5. Finalmente se incluye una capa de aplanamiento y una capa densa con un tamaño de 5 neuronas (en la arquitectura original son 7 pero debido a la reducción de clases que se realizó ahora es 5). En la siguiente imagen se ilustra la arquitectura elegida. 

![Arquitectura](/images/architecture.png)

Los conceptos clave de esta arquitectura son los siguientes:
1. **5 Bloques Convolucionales**: El modelo está compuesto por cinco bloques, donde cada bloque realiza ciertas operaciones específicas.
2. **Capas Convolucionales (Convolutional Layers)**: Cada bloque contiene dos capas convolucionales que se encargan de aplicar filtros para extraer características de la imagen.
3. **Filtros Convolucionales**: Son matrices de 4x4 que se usan para convolver la imagen y detectar patrones locales como bordes, texturas, etc.
4. **Inicialización con "Xavier Weight Initialization"**: Método de inicialización de los pesos de los filtros, diseñado para mantener la varianza de los activaciones a través de las capas, promoviendo una convergencia más rápida y estable durante el entrenamiento.
5. **Capa de Normalización de Batches (Batch Normalization Layer)**: Normaliza las activaciones de la capa anterior, estabilizando y acelerando el proceso de entrenamiento al mantener la media y la varianza constantes.
6. **Dropout de 0.5**: Técnica de regularización que desactiva aleatoriamente el 50% (0.5) de las neuronas durante el entrenamiento, ayudando a prevenir el sobreajuste (overfitting).
7. **Capa de Aplanamiento (Flatten Layer)**: Transforma las matrices multidimensionales de las capas convolucionales en un vector unidimensional, preparándolo para la capa densa.
8. **Capa Densa**: La capa final del modelo tiene cinco neuronas, que corresponden a las cinco clases de expresiones faciales a reconocer. Originalmente eran siete neuronas (para siete clases) pero se redujeron a cinco debido a una reducción en el número de clases.

Para compilar y entrenar el modelo se utilizaron 50 epochs con un tamaño de batch de 64. Los hiperparámetros utilizados, también obtenidos del artículo de investigación, son los siguientes:

| Hiperparámetro | Valor                      |
| -------------- | -------------------------- |
| Loss           | Categorical Crossentropy   |
| Optimizer      | Adam                       | 
| Learning rate  | 0.001                      |
| Weight decay   | 0.000001                   |
| momentum       | 0.9                        |

### Loss (Función de Pérdida):
- Categorical Crossentropy: Es una función de pérdida utilizada para problemas de clasificación multiclase. Mide la diferencia entre la distribución de probabilidad predicha y la distribución real de las clases.
### Optimizer (Optimizador):
- Adam: Es un algoritmo de optimización adaptativa que ajusta los parámetros del modelo basándose en estimaciones de momentos de primer y segundo orden (medias y varianzas de los gradientes). Es conocido por su eficiencia y efectividad en el entrenamiento de redes neuronales.
### Learning Rate (Tasa de Aprendizaje):
- 0.001: La tasa de aprendizaje es un hiperparámetro que controla el tamaño de los pasos que da el optimizador al actualizar los parámetros del modelo. Un valor de 0.001 indica que los ajustes a los pesos del modelo serán pequeños en cada actualización, permitiendo un aprendizaje más suave y estable.
### Weight Decay (Decaimiento de los Pesos):
- 0.000001: El weight decay es una técnica de regularización que añade un término de penalización a la función de pérdida proporcional a la magnitud de los pesos. Esto ayuda a prevenir el sobreajuste al restringir el crecimiento excesivo de los pesos.
### Momentum (Momento):
- 0.9: El momentum es una técnica utilizada en optimización para acelerar el entrenamiento de redes neuronales. Ayuda a suavizar las actualizaciones de los parámetros acumulando una fracción de las actualizaciones pasadas (0.9 en este caso) y usándolas para actualizar los parámetros actuales. Esto ayuda a evitar oscilaciones y a mantener el modelo en la dirección correcta hacia el mínimo de la función de pérdida.



# Diagnóstico de resultados iniciales

## Resultados
|                 | Train           | Validation       | Test          |
| --------------  | --------------- | ---------------- | ------------- |
| accuracy        | 0.6782          | 0.6592           | 0.6658        |
| loss            | 0.8462          | 0.9161           | 0.8900        |

Con respecto a los resultados obtenidos en el artículo de referencia, los resultados obtenidos son buenos. Ya que se acercan al 69% de accuracy obtenido en el estado del arte. Sin embargo, los resultados siguen teniendo un porcentaje de accuracy pequeño y difieren en un 3% con los del estado del arte. Aún así se puede notar en la siguiente imagen que el accuracy y el loss tanto en train como validation tuvo un buen comportamiento durante los epochs.
![Acc_Loss](/images/acc_loss_grafs.png)

Se generó una matriz de confusión para obtener la cantidad de predicciones verdaderas, falsos positivos y falsos negativos de cada clase. La matriz es la siguiente:
![Mátriz](/images/matrixog.png)

De acuerdo con la matriz, la clase que tuvo un mejor número de predicciones correctas fue la clase "happy", seguido de la clase "neutral", la clase "angry", la clase "sad" y la clase "fearful". Por otro lado, se nota que todas las clases (exceptuando la clase "happy"), tienen un alto número de predicciones incorrectas. Por ejemplo, las clases "sad" y "angry" se confunden con la clase "neutral" y la clase "fearful" se confunde con "angry", "neutral" y "sad".

Utilizando esta matriz se obtuvieron las siguientes métricas:

### Precisión (Precision):
- La precisión es la proporción de verdaderos positivos (𝑇𝑃) entre todos los ejemplos que el modelo ha etiquetado como positivos. En otras palabras, mide la exactitud de las predicciones positivas del modelo. Una alta precisión indica que el modelo comete pocos errores al etiquetar ejemplos como positivos (es decir, pocos falsos positivos).

### Recall (Sensibilidad o Tasa de Verdaderos Positivos):
- El recall es la proporción de verdaderos positivos entre todos los ejemplos que son realmente positivos. Mide la capacidad del modelo para identificar todos los ejemplos positivos. Un alto recall indica que el modelo es capaz de encontrar la mayoría de los ejemplos positivos (es decir, pocos falsos negativos).

### F1-Score
El F1-score es la media armónica de la precisión y el recall. Es una métrica combinada que equilibra ambos aspectos, proporcionando una única medida del rendimiento del modelo cuando se consideran igualmente importantes tanto la precisión como el recall. Un alto F1-score indica que el modelo tiene un buen equilibrio entre precisión y recall. Es especialmente útil cuando se necesita una medida única del rendimiento del modelo en problemas de clasificación con una distribución de clases desequilibrada.

En cuanto a los resultados de precision, recall y f1-score, se obtuvo lo siguiente:
![Reporte](/images/classification_report.png)

Se puede notar que la clase "happy" tiene un puntaje mucho mayor que las otras clases en todas las métricas. Esto puede ser debido a que se cuentan con más datos en la clase "happy". Por otro lado, la clase "sad" cuenta con un puntaje bajo donde tanto la precision, el recall y el f1-score son de entre 0.53 y 0.54, lo que representa una diferencia de aproximadamente 0.3 con la clase "happy".

Debido a los resultados bajos en accuracy y en las métricas de recall, precision y f1-score, se puede concluir que falta entrenar al modelo o que el modelo tiene **underfitting**. Esto debido a que hay un gran número de predicciones para cada clase que no fueron las correctas. Esto se puede deber a una o varias de las siguientes opciones:

1. Desbalance de Clases:
La clase happy tiene un mayor número de imágenes que las otras clases (con una diferencia entre 2000 y 3000 aproximadamente), es decir, el modelo tiene muchas más muestras de una clase que de las otras, por lo que el modelo podría estar aprendiendo a predecir siempre la clase mayoritaria. Esto puede resultar en una alta cantidad de verdaderos positivos (TP) pero también en una alta cantidad de falsos positivos (FP) y falsos negativos (FN), lo cual afectará negativamente las métricas de precisión, recall y F1 score.

2. Errores en el Preprocesamiento de Datos:
Los datos deben estar correctamente preprocesados para asegurar que el modelo pueda aprender de ellos de manera efectiva. Agregar parámetros en el ImagedataGenerator podría ayudar a generalizar más los datos y que el modelo tenga un mejor comportamiento con datos que nunca ha utilizado.

# Mejora del modelo
Para mejorar el modelo se realizó lo siguiente:
### Modificación de dropout
Se decidió modificar el dropout de 0.5 a 0.4 para desactivar de forma aleatoria solamente el 40% de las neuronas. Este decremento de solo 10% fue para que el riesgo de sobreajuste no aumentará considerablemente y se mejorara la capacidad del modelo para generalizar a nuevos datos.
### Modificación de learning rate
Aunque se mantuvo el learning rate inicial de 0.001, se utilizó el callback ReduceLROnPlateau, el cual monitoreó el valor de accuracy en validation set. En caso de que el validation accuracy no mejorará en 6 epochs consecutivas, se multiplicaba el learning rate actual por un factor de 0.1. En este caso, en el epoch 59 el learning rate se redujo a 0.0001 y en el epoch 86 se redujo a 0.0001.
### Modificación de epochs
Se aumentó el número de epochs de 50 a 100 y se utilizó el callback EarlyStopping. Si el validation accuracy no mejoraba en 10 epochs consecutivas se detenía el entrenamiento. En este caso, esto sucedió en el epoch 90.
### Checkpoint
Se utilizó un checkpoint para guardar el modelo con mejor validation accuracy.
### Modificación de ImageDataGenerator
Se agregaron los parámetros de widht shift y height shift al train generator y se aumentó el batch de 64 a 128, esto para que el modelo tuviera una mejor generalización de datos y se comportara mejor con datos que no conoce.
### Combinación de dataset
Se descubrió que el dataset elegido tiene una gran cantidad de imágenes que están mal clasificadas lo que genera un alto nivel de ruido. Por lo tanto, se decidió combinar el data set CK+ obtenido de [Kaggle](https://www.kaggle.com/datasets/shuvoalok/ck-dataset), que a pesar de ser pequeño no cuenta con ruido. Esto se realizó con la intención de que el modelo tuviera una mayor cantidad de datos etiquetados correctamente. Se agregaron 391 imágenes en train, 94 en test y 88 en validation.

Los resultados de accuracy y loss en validation, train y test fueron los siguientes:

|                 | Train           | Validation       | Test          |
| --------------  | --------------- | ---------------- | ------------- |
| accuracy        | 0.7213          | 0.6822           | 0.6817        |
| loss            | 0.7161          | 0.8804           | 0.8728        |

Comparado con el modelo original, test y validation tuvieron aproximadamente 2% mejor accuracy y loss. 

De igual manera, la matriz de confusión arrojó mejores resultados.
![Mátriz](/images/matrix.png)
Se puede notar que en la mayoría de las clases se obtuvo un mayor número de predicciones correctas. Para cada clase se obtuvieron estos resultados:

### Angry:
Las predicciones correctas (angry → angry) disminuyeron de 484 a 476.
Las confusiones con "fearful" disminuyeron de 53 a 37.
Las confusiones con "happy" aumentaron de 28 a 47.
Las confusiones con "neutral" disminuyeron de 128 a 111.
Las confusiones con "sad" aumentaron de 72 a 94.

### Fearful:
Las predicciones correctas (fearful → fearful) disminuyeron de 380 a 279.
Las confusiones con "angry" aumentaron ligeramente de 112 a 124.
Las confusiones con "happy" aumentaron levemente de 33 a 35.
Las confusiones con "neutral" aumentaron de 109 a 128.
Las confusiones con "sad" aumentaron considerablemente de 147 a 215.

### Happy:
Las predicciones correctas (happy → happy) aumentaron ligeramente de 1223 a 1227.
Las confusiones con las demás emociones disminuyeron en general.

### Neutral:
Las predicciones correctas (neutral → neutral) disminuyeron de 691 a 675.
Las confusiones con "angry" y "fearful" aumentaron levemente.
Las confusiones con "happy" aumentaron de 75 a 78.
Las confusiones con "sad" aumentaron de 88 a 107.

### Sad:
Las predicciones correctas (sad → sad) aumentaron de 483 a 498.
Las confusiones con "angry" disminuyeron de 104 a 100.
Las confusiones con "fearful" disminuyeron de 82 a 49.
Las confusiones con "happy" aumentaron ligeramente de 44 a 50.
Las confusiones con "neutral" aumentaron de 214 a 230.

En resumen, el modelo mejoró ligeramente en la predicción de "happy" y "sad", pero empeoró en las demás emociones, especialmente en "fearful" donde aumentaron significativamente las confusiones con "sad". En general, el modelo parece tener más dificultad distinguiendo entre algunas emociones en la segunda matriz.

Estos resultados se ven reflejados en los resultados de recall, precision y f1-score.
![ClassificationReport](/images/cr.png)

Mejoras:

La precisión (precision) para "angry" mejoró de 0.60 a 0.62.
El recall para "angry" mejoró ligeramente de 0.62 a 0.63.
El recall para "fearful" mejoró significativamente de 0.36 a 0.49.
La precisión para "happy" mejoró de 0.85 a 0.87.
El recall para "happy" se mantuvo igual en 0.89.
El recall para "neutral" mejoró de 0.72 a 0.74.
La precisión para "sad" mejoró de 0.53 a 0.60.
La precisión (accuracy), el promedio macro (macro avg) y el promedio ponderado (weighted avg) mejoraron ligeramente.

Empeoramientos:

La precisión para "fearful" empeoró de 0.68 a 0.65.
El f1-score para "fearful" empeoró de 0.47 a 0.46.
El f1-score para "neutral" empeoró ligeramente de 0.63 a 0.64.
El recall para "sad" empeoró significativamente de 0.54 a 0.52.
El f1-score para "sad" empeoró de 0.53 a 0.56.

Hubo mejoras en la mayoría de las métricas, especialmente en el recall de "fearful" y la precisión de "sad". Sin embargo, también se observan algunos empeoramientos, sobre todo en el recall de "sad". En general, parece haber una mejora ligera en el desempeño global según las métricas de accuracy y los promedios.


Aunque el test accuracy aumentó y el test loss disminuyó con respecto al modelo original, se puede notar que en estos resultados existe **overfitting** ya que la diferencia de loss entre test y train es de 16%. De acuerdo con la siguiente gráfica se puede decir que existe **overfitting**.

![Gráfica](/images/ref_graf.jpeg)

Aún así los resultados obtenidos se acercan más a los del estado del arte. Se acerca all accuracy de 0.69 obtenido por Bodapati y Strilakshmi en el artículo de investigación "A Deep CNN Architecture for Facial Expression Recognition in the Wild." Además, de acuerdo con Amil Khanzada, Charles Bai y Ferhat Turker Celepcikay en el artículo "Facial Expression Recognition with Deep Learning" debido al ruido del dataset solo es posible superar el 0.70 de accuracy utilizando técnicas de registro de puntos faciales antes de realizar la clasificación de imágenes o bien utilizar modelos de transfer learning pre entrenados como ResNet50, VGG16 y SeNet50 que requieren una mayor capacidad de cómputo qu ela disponible.

Cabe mencionar que el modelo tuvo un buen comportamiento con imágenes de internet. Los resultados obtenidos con queries se encuentran en Testing.ipynb y fueron los siguientes.

![Queries](/images/queries.png)

![Queries](/images/queries2.png)



# Referencias
1. P. Verma, V. Aggrawal, and J. Maggu, "FExR.A-DCNN: Facial Emotion Recognition with Attention Mechanism using Deep Convolution Neural Network," in *Proc. 2022 Fourteenth Int. Conf. Contemporary Computing (IC3-2022)*, New York, NY, USA, 2022, pp. 196–203. doi: 10.1145/3549206.3549243.
2. Bodapati, J.D., Srilakshmi, U. & Veeranjaneyulu, N. FERNet: A Deep CNN Architecture for Facial Expression Recognition in the Wild. J. Inst. Eng. India Ser. B 103, 439–448 (2022). https://doi.org/10.1007/s40031-021-00681-8
3. A. Ananthu, "Emotion Detection FER Dataset," Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer?select=train. [Accessed: 19-May-2024].
4. S. P. Shuvo, "split_folders for train test val split of images," Kaggle, 2023. [Online]. Available: https://www.kaggle.com/code/shuvostp/split-folders-for-train-test-val-split-of-images. [Accessed: 19-May-2024].
5. Khanzada, A., Bai, C., Celepcikay, F. T., & Khosla, A. (2018). Facial expression recognition with deep learning. Stanford University. https://cs231n.stanford.edu/reports/2016/pdfs/023_Report.pdf 
6. "CK+ dataset" Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer?select=train. [Accessed: 4-June-2024].


