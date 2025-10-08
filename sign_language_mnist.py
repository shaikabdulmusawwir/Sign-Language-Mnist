import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import keras as kf
import matplotlib.pyplot as plt
import os

# Use local relative paths (Windows-friendly)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "sign_mnist_train.csv")
VALID_PATH = os.path.join(BASE_DIR, "data", "sign_mnist_test.csv")

# Constants
NO_CLASSES = 26

# Load datasets
trainData = pd.read_csv(TRAIN_PATH)
validData = pd.read_csv(VALID_PATH)

# Split data
x_train = np.array(trainData.drop(columns=['label']))
y_train = np.array(trainData['label'])
x_valid = np.array(validData.drop(columns=['label']))
y_valid = np.array(validData['label'])

# Reshape and normalize
x_train = x_train.reshape(len(x_train), 28, 28, 1) / 255.0
x_valid = x_valid.reshape(len(x_valid), 28, 28, 1) / 255.0

print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

# CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(15, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

# Compile and train
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=7, validation_data=(x_valid, y_valid))

# Save the model
model.save('sign_language_mnist_cnn.h5')

# Test and visualize
testImage = x_valid[10]
prediction = model.predict(testImage.reshape(-1, 28, 28, 1))

plt.imshow(testImage.reshape(28, 28))
plt.xlabel(f"Prediction: {np.argmax(prediction)}, Actual Value: {y_valid[10]}")
plt.show()
