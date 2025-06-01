import numpy as np
import tensorflow as tf
import data
import model
import json

def train(_model, epochs=20, save_path="./model.keras"):
    history = []
    validation_history = []
    for i in range(epochs):
        history.append(_model.fit(data.train_dataset, verbose=1))
        validation_history.append(_model.evaluate(data.val_dataset, verbose=1))
        _model.save(save_path)
    return history, validation_history

if __name__ == "__main__":
    epochs = int(input("How many epochs? "))
    save_path = input("Where to save the model? ")
    history, validation_history = train(model.build_model(), epochs=epochs, save_path=save_path)
    with open(save_path + ".history", "w") as history_file:
        history_file.write(json.dumps(history))
    with open(save_path + ".validation-history", "w") as val_history_file:
        val_history_file.write(json.dumps(validation_history))
    print("Done")