# Clasificación de imágenes de expresiones faciales con CNN
# Conjunto de datos inicial
El conjunto de datos utilizado para este proyecto se obtuvo de [Kaggle](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)

Este conjunto de datos contiene 35,685 imágenes en escala de grises de rostros humanos con diferentes expresiones faciales. Las imágenes tienen un tamaño de 48x48 píxeles. 


# División de conjunto de datos
Los datos se dividen en dos subconjuntos: train y test, con una proporción de 80% para entrenamiento y 20% para evaluación. Cada subconjunto se compone de siete carpetas, cada una de las cuales representa una clase de expresión facial:


| Nombre carpeta | Test (cantidad de imágenes) | Train (cantidad de imágenes) |
| -------------- | --------------------------- | ---------------------------- |
| angry          | 958                         | 3995                         |
| disgusted      | 111                         | 436                          |
| fearful        | 1024                        | 4097                         |
| happy          | 1774                        | 7215                         |
| neutral        | 1233                        | 4965                         |
| sad            | 1247                        | 4830                         |
| surprised      | 831                         | 3171                         |



Es importante mencionar que la división del conjunto de datos es crucial para que se pueda entrenar el modelo (datos en train) y posteriormente evaluar su rendimiento con datos reales (datos en test). Hacer esta división también evita el sobreentrenamiento (overfitting), el cual ocurre cuando un modelo se ajusta demasiado a los datos de entrenamiento y no puede generalizarse bien a nuevos datos.
