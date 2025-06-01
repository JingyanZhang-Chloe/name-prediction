import keras
from keras import layers

def block(x, filters, name=None):
    x = layers.Conv2D(filters, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)   #randomly forget 10% of the output of the previous layer
    shortcut = x
    x = layers.Conv2D(filters, kernel_size=3, padding='same', activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.add([shortcut, x])
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    return x

def build_model():
    model_input = layers.Input((120, 150, 3))
    x = block(model_input, 64)
    x = block(x, 128)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(64, kernel_size=1, padding='valid', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(100, activation="softmax")(x)
    model = keras.Model(model_input, x)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

"""def build_model():
    model_input = layers.Input((120, 150, 3))
    x = block(model_input, 64)
    x = block(x, 128)
    x = block(x, 256)
    x = layers.Conv2D(64, kernel_size=1, padding='valid', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(100, activation="softmax")(x)
    model = keras.Model(model_input, x)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model"""