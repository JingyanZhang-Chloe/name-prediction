import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

dataset = tf.keras.utils.image_dataset_from_directory(
    directory='data/',
    labels='inferred',              
    label_mode='int',               
    batch_size=32,                  
    shuffle=True                    
)