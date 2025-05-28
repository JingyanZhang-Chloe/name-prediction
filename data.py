import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


dataset = tf.keras.utils.image_dataset_from_directory(
    directory='dataset/',
    labels='inferred',              # Automatically use folder names as labels
    label_mode='int',               # Can be 'int', 'categorical', or 'binary'
    image_size=(224, 224),          # Resize images to this size
    batch_size=32,                  # Number of images per batch
    shuffle=True                    # Shuffle the data
)

class_names = dataset.class_names
print(class_names)

for images, labels in dataset.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        ax = plt.subplot(2, 4, i + 1)  # 2 rows, 4 columns
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")