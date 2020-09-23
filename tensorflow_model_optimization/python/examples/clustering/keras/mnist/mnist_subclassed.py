# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=missing-docstring
"""Train a simple convnet that is written as a keras subclassed model
on the MNIST dataset and cluster it.

This example is based on the sample that can be found here:
https://www.tensorflow.org/tutorials/quickstart/advanced
"""

from __future__ import print_function

import tensorflow as tf
import os

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from tensorflow_model_optimization.python.core.clustering.keras import cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.clustering.keras import clustering_callbacks

BATCH_SIZE = 32
EPOCHS = 5
EPOCHS_FINE_TUNING = 4

# Load and prepare MNIST dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

# Use tf.data to batch and shuffle the dataset.
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

# Build the model using the Keras model subclassing API.
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()

# Choose an optimizer and loss function for training.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

# Select metrics to measure the loss and the accuracy of the model. 
# These metrics accumulate the values over epochs and then print the overall result.
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# Use tf.GradientTape to train the model as it is done in the tutorial.
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

# Test the model.
@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result()}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result()}'
  )

def cluster_model(model, x_train, y_train, x_test, y_test):
  print('Clustering model')

  clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': cluster_config.CentroidInitialization.DENSITY_BASED
  }

  # Cluster model
  clustered_model = cluster.cluster_weights(model, **clustering_params)

  # Use smaller learning rate for fine-tuning
  # clustered model
  opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

  clustered_model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = opt,
    metrics = ['accuracy'])

  # Fine-tune clustered model
  clustered_model.fit(
      x_train,
      y_train,
      batch_size = BATCH_SIZE,
      epochs = EPOCHS_FINE_TUNING,
      verbose = 1,
      validation_split = 0.1)

  score = clustered_model.evaluate(x_test, y_test, verbose=0)
  print('Clustered model test loss:', score[0])
  print('Clustered model test accuracy:', score[1])

  return clustered_model

def test_clustered_model(clustered_model, x_test, y_test):
  # Stripping the model
  stripped_model = cluster.strip_clustering(clustered_model)
  stripped_model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = 'adam',
    metrics = ['accuracy'])

  # Checking that the stripped model's accuracy matches the clustered model
  score = stripped_model.evaluate(x_test, y_test, verbose=0)

  print('Stripped model test loss:', score[0])
  print('Stripped model test accuracy:', score[1])

  # Checking that we have the number of weights less than the
  # number of clusters.
  for layer in stripped_model.layers:
    nr_unique_weights = len(set(layer.get_weights()[0].flatten())) \
      if len(layer.get_weights()) > 0 else 0
    print("Layer name: {}, number of clusters: {}".format(
      layer.name, nr_unique_weights
    ))

# Cluster and fine-tune model
clustered_model = cluster_model(model, x_train, y_train, x_test, y_test)

# Test clustered model (strip clustering)
test_clustered_model(clustered_model, x_test, y_test)
