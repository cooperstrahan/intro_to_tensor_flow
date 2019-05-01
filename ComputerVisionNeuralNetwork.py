from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from tensorflow import keras
print(tf.version)

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(training_images[55])

# print(training_labels[55])
# print(training_images[55])

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)