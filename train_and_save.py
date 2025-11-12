# train_and_save.py
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

MODEL_PATH = "mnist_model.h5"

def build_model():
    model = keras.Sequential([
        keras.Input(shape=(784,)),
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 784)).astype("float32") / 255.0
    test_images = test_images.reshape((10000, 784)).astype("float32") / 255.0
    return (train_images, train_labels), (test_images, test_labels)

def train_and_save():
    (train_images, train_labels), (test_images, test_labels) = load_data()
    model = build_model()
    model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_split=0.1)
    loss, acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"✅ Test accuracy: {acc:.4f}, loss: {loss:.4f}")
    model.save(MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print(f"{MODEL_PATH} already exists. Delete it if you want to retrain.")
    else:
        train_and_save()
