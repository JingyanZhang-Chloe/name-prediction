import keras
from keras import layers
import tensorflow as tf

def block(x, filters, name=None):
    x = layers.Conv2D(filters, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    shortcut = x
    x = layers.Conv2D(filters, kernel_size=3, padding='same', activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.add([shortcut, x])
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    return x

def build_model():
    model_input = layers.Input((120, 150, 3))
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(100, activation="softmax")(x)
    model = keras.Model(model_input, x)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

keras.utils.plot_model(build_model(), to_file='./model_plot.png')