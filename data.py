from logging.config import valid_ident
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

train_dataset = tf.keras.utils.image_dataset_from_directory(
    directory='data/',
    validation_split = 0.2,
    subset = 'training',
    seed = 123,
    labels='inferred',              
    label_mode='int',               
    batch_size=32,           
    shuffle=True                 
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    directory='data/',
    validation_split = 0.2,
    subset = 'validation',
    seed = 123,
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
    layers.Rescaling(1./255),
])

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
test_dataset = val_dataset.map(lambda x, y: (data_augmentation(x), y))
