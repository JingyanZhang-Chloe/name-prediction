import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

dataset = tf.keras.utils.image_dataset_from_directory(
    directory='data/',
    labels='inferred',              
    label_mode='int',               
    batch_size=32,                  
    shuffle=True                    
)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

augmented_dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
