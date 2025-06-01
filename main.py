import numpy as np
import tensorflow as tf
import data
import model
import json

def train(_model, epochs=20, save_path="./model.keras"):
    history = []
    validation_history = []
    for i in range(epochs):
        history.append(_model.fit(data.train_dataset, verbose=1).history)
        validation_history.append(_model.evaluate(data.val_dataset, verbose=1))
        _model.save(save_path)
    return history, validation_history

if __name__ == "__main__":
    epochs = int(input("How many epochs? "))
    save_path = input("Where to save the model? ")
    _model = model.build_model()
    _model.summary()
    history, validation_history = train(_model, epochs=epochs, save_path=save_path)
    print("Training history")
    print(history)
    print("Validation history")
    print(validation_history)
    print("egg")