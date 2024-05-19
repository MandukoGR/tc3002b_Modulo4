
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = './data/train'
augmented_path = './data/augmented'


train_datagen = ImageDataGenerator(
							rescale = 1./255,
							rotation_range = 180,
                            brightness_range= (0.1, 0.9),
							zoom_range = 0.2,
							horizontal_flip = True,)

train_generator = train_datagen.flow_from_directory(
							train_path,
							target_size = (40, 40),
							batch_size = 10,
							class_mode ='categorical',
                        	save_to_dir= augmented_path,
              				save_prefix='aug',
              				save_format='png')
classes = {
    0: 'angry',
    1: 'fearful',
    2: 'happy',
    3: 'neutral',
    4: 'sad',
}

images, labels = train_generator[0]