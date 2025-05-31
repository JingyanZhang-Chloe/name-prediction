import keras 
import data

my_model = keras.Sequential([
    keras.layers.Conv2D(64, padding="same", kernel_size=3, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.MaxPooling2D(pool_size=2, strides=2),
    
    keras.layers.Conv2D(128, padding="same", kernel_size=3, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.MaxPooling2D(pool_size=2, strides=2),
    
    keras.layers.Conv2D(256, padding="same", kernel_size=5, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(100, activation="softmax")
])

# Compile the model
my_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

my_model.fit(data.train_dataset, epochs=5, verbose="1")

my_model.save("./leo_model.keras")