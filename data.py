from logging.config import valid_ident
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import layers

train_dataset = keras.utils.image_dataset_from_directory(
    directory='data/',
    validation_split = 0.2,
    subset = 'training',
    seed = 123,
    labels='inferred',              
    label_mode='int',               
    batch_size=32,           
    shuffle=True,
    image_size=(120, 150)
)

val_dataset = keras.utils.image_dataset_from_directory(
    directory='data/',
    validation_split = 0.2,
    subset = 'validation',
    seed = 123,
    labels='inferred',              
    label_mode='int',               
    batch_size=32,           
    shuffle=True,
    image_size=(120, 150)           
)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.Rescaling(1./127.5, offset=-1),
])

class_names = train_dataset.class_names

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
test_dataset = val_dataset.map(lambda x, y: (data_augmentation(x), y))
