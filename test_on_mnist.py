import tensorflow as tf
import keras
from keras.datasets import mnist
import numpy as np

(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()

X_train_mnist = X_train_mnist[..., tf.newaxis]  # shape: (60000, 28, 28, 1)
X_test_mnist = X_test_mnist[..., tf.newaxis]

X_train_mnist = tf.image.resize(X_train_mnist, size=(120, 150))
X_test_mnist = tf.image.resize(X_test_mnist, size=(120, 150))

X_train_mnist = tf.image.grayscale_to_rgb(X_train_mnist)
X_test_mnist = tf.image.grayscale_to_rgb(X_test_mnist)

print(F"shape for X_train : {X_train_mnist.shape} and shape for X_test : {X_test_mnist.shape}")

def test(model, X_test=X_test_mnist, y_test=y_test_mnist):
    # Train the model
    model.fit(X_train_mnist, y_train_mnist, epochs=10 , batch_size=32)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)

    # Show predictions on first 5 samples
    for i, sample in enumerate(X_test[:5]):
        # Add batch dimension: (120, 150, 3) → (1, 120, 150, 3)
        sample_batch = tf.expand_dims(sample, axis=0)
        pred_probs = model.predict(sample_batch)
        pred_class = np.argmax(pred_probs)

        print(f"Predicted: {pred_class}, True: {y_test[i]}")