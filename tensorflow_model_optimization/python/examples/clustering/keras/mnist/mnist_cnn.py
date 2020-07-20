# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Train a simple convnet on the MNIST dataset and cluster it.

This example is based on the sample that can be found here:
https://www.tensorflow.org/model_optimization/guide/quantization/training_example
"""

from __future__ import print_function
import datetime
import os

from absl import app as absl_app
from absl import flags

import tensorflow as tf
from tensorflow_model_optimization.python.core.clustering.keras import cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.clustering.keras import clustering_callbacks

keras = tf.keras

FLAGS = flags.FLAGS

batch_size = 128
epochs = 12
epochs_fine_tuning = 4

flags.DEFINE_boolean('enable_eager', True, 'Trains in eager mode.')
flags.DEFINE_string('output_dir', '/tmp/mnist_train/',
                    'Output directory to hold tensorboard events')


def load_mnist_dataset():
  mnist = keras.datasets.mnist
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

  # Normalize the input image so that each pixel value is between 0 to 1.
  train_images = train_images / 255.0
  test_images = test_images / 255.0

  return (train_images, train_labels), (test_images, test_labels)

def build_sequential_model():
  "Define the model architecture."

  return keras.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28)),
    keras.layers.Reshape(target_shape=(28, 28, 1)),
    keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
  ])


def train_model(model, x_train, y_train, x_test, y_test):
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer='adam',
      metrics=['accuracy'])

  # Print the model summary.
  model.summary()

  # Model needs to be clustered after initial training
  # and having achieved good accuracy
  model.fit(
      x_train,
      y_train,
      batch_size=batch_size,
      epochs=epochs,
      verbose=1,
      validation_split=0.1)

  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  
  return model


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
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  optimizer=opt,
  metrics=['accuracy'])

  # Add callback for tensorboard summaries
  log_dir = os.path.join(
      FLAGS.output_dir,
      datetime.datetime.now().strftime("%Y%m%d-%H%M%S-clustering"))
  callbacks = [
      clustering_callbacks.ClusteringSummaries(
          log_dir,
          cluster_update_freq='epoch',
          update_freq='batch',
          histogram_freq=1)
  ]

  # Fine-tune clustered model
  clustered_model.fit(
      x_train,
      y_train,
      batch_size=batch_size,
      epochs=epochs_fine_tuning,
      verbose=1,
      callbacks=callbacks,
      validation_split=0.1)

  score = clustered_model.evaluate(x_test, y_test, verbose=0)
  print('Clustered model test loss:', score[0])
  print('Clustered model test accuracy:', score[1])

  return clustered_model


def test_clustered_model(clustered_model, x_test, y_test):
  # Ensure accuracy persists after serializing/deserializing the model
  clustered_model.save('clustered_model.h5')
  # To deserialize the clustered model, use the clustering scope
  with cluster.cluster_scope():
    loaded_clustered_model = keras.models.load_model('clustered_model.h5')

  # Checking that the deserialized model's accuracy matches the clustered model
  score = loaded_clustered_model.evaluate(x_test, y_test, verbose=0)
  print('Deserialized model test loss:', score[0])
  print('Deserialized model test accuracy:', score[1])

  # Ensure accuracy persists after stripping the model
  stripped_model = cluster.strip_clustering(loaded_clustered_model)
  stripped_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

  # Checking that the stripped model's accuracy matches the clustered model
  score = stripped_model.evaluate(x_test, y_test, verbose=0)
  print('Stripped model test loss:', score[0])
  print('Stripped model test accuracy:', score[1])


def main(unused_argv):
  if FLAGS.enable_eager:
    print('Running in Eager mode.')
    tf.compat.v1.enable_eager_execution()

  # the data, shuffled and split between train and test sets
  (x_train, y_train), (x_test, y_test) = load_mnist_dataset()

  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')

  # Build model
  model = build_sequential_model()
  # Train model
  model = train_model(model, x_train, y_train, x_test, y_test)
  # Cluster and fine-tune model
  clustered_model = cluster_model(model, x_train, y_train, x_test, y_test)
  # Test clustered model (serialize/deserialize, strip clustering)
  test_clustered_model(clustered_model, x_test, y_test)


if __name__ == '__main__':
  absl_app.run(main)
