# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import print_function

import os

from absl import app as absl_app
import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.clustering.keras import cluster as tfmot_cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster_config as tfmot_cluster_config
from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras.collab_opts.cluster_preserve import (
    default_8bit_cluster_preserve_quantize_scheme,)
from tensorflow_model_optimization.python.core.quantization.keras.collab_opts.cluster_preserve.cluster_utils import (
    strip_clustering_cqat,)

layers = tf.keras.layers


def setup_model(input_shape, image_train, label_train):
  """Baseline model."""
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3),
                             activation=tf.nn.relu),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10)
  ])
  compile_and_fit(model, image_train, label_train, 5)

  return model


def _get_callback(model_dir):
  """Create callbacks for Keras model training."""
  check_point = tf.keras.callbacks.ModelCheckpoint(
      save_best_only=True,
      filepath=os.path.join(model_dir, 'model.ckpt-{epoch:04d}'),
      verbose=1)
  tensorboard = tf.keras.callbacks.TensorBoard(
      log_dir=model_dir, update_freq=100)
  return [check_point, tensorboard]


def compile_and_fit(model,
                    image_train,
                    label_train,
                    epochs):
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
  )

  callbacks_to_use = _get_callback(model_dir='./logs')
  model.fit(
      image_train,
      label_train,
      batch_size=500,
      validation_split=0.1,
      epochs=epochs,
      callbacks=callbacks_to_use,
      verbose=1)


def cluster_model(model, train_images, train_labels):
  """Apply the clustering wrapper, compile and train the model."""
  clustering_params = {
      'number_of_clusters': 16,
      'cluster_centroids_init':
      tfmot_cluster_config.CentroidInitialization.DENSITY_BASED,
  }
  model = tfmot_cluster.cluster_weights(
      model, **clustering_params)
  model.summary()
  compile_and_fit(model,
                  train_images,
                  train_labels,
                  1)

  return model


def cluster_preserve_quantize_model(clustered_model,
                                    train_images,
                                    train_labels):
  """Cluster-preserve QAT model."""
  quant_aware_annotate_model = (
      quantize.quantize_annotate_model(clustered_model))
  quant_aware_model = quantize.quantize_apply(
      quant_aware_annotate_model,
      scheme=default_8bit_cluster_preserve_quantize_scheme
      .Default8BitClusterPreserveQuantizeScheme())
  quant_aware_model.summary()
  compile_and_fit(quant_aware_model,
                  train_images,
                  train_labels,
                  1)

  return quant_aware_model


def evaluate_model_fp32(model, image_test, label_test):
  score = model.evaluate(image_test, label_test, verbose=0)
  return score[1]


def print_unique_weights(model):
  """Check Dense and Conv2D layers."""
  for layer in model.layers:
    if (isinstance(layer, tf.keras.layers.Conv2D)
        or isinstance(layer, tf.keras.layers.Dense)
        or isinstance(layer, quantize.quantize_wrapper.QuantizeWrapper)):
      for weights in layer.trainable_weights:
        np_weights = tf.keras.backend.get_value(weights)
        unique_weights = len(np.unique(np_weights))
        if isinstance(layer, quantize.quantize_wrapper.QuantizeWrapper):
          print(layer.layer.__class__.__name__, ' (', weights.name,
                ') unique_weights: ', unique_weights)
        else:
          print(layer.__class__.__name__, ' (', weights.name,
                ') unique_weights: ', unique_weights)


# This code is directly from
# https://www.tensorflow.org/model_optimization/guide/quantization/training_example
def evaluate_model(interpreter, test_images, test_labels):
  input_index = interpreter.get_input_details()[0]['index']
  output_index = interpreter.get_output_details()[0]['index']

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for i, test_image in enumerate(test_images):
    if i % 1000 == 0:
      print('Evaluated on {n} results so far.'.format(n=i))
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)
    # Run inference.
    interpreter.invoke()
    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  print('\n')
  # Compare prediction results with ground truth labels to calculate accuracy.
  prediction_digits = np.array(prediction_digits)
  accuracy = (prediction_digits == test_labels).mean()
  return accuracy


def main(unused_args):
  # Load the MNIST dataset.
  mnist = tf.keras.datasets.mnist
  # Shuffle and split data to generate training and testing datasets
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
  # Normalize the input images so that each pixel value is between 0 and 1.
  train_images = train_images / 255.0
  test_images = test_images / 255.0

  input_shape = (28, 28)
  # Create and train the baseline model
  model = setup_model(input_shape, train_images, train_labels)
  # Apply clustering API and retrain the model
  clustered_model = cluster_model(model, train_images, train_labels)
  print('Apply clustering:')
  clst_acc = evaluate_model_fp32(clustered_model, test_images, test_labels)
  clustered_model_stripped = tfmot_cluster.strip_clustering(clustered_model)
  print('Apply cluster-preserve quantization aware training (cqat):')
  # Start from pretrained clustered model, apply CQAT API, retrain the model
  cqat_model = cluster_preserve_quantize_model(
      clustered_model_stripped,
      train_images,
      train_labels
      )
  cqat_acc = evaluate_model_fp32(cqat_model, test_images, test_labels)
  # This only removes extra variables introduced by clustering
  # but the quantize_wrapper stays
  cqat_model_stripped = strip_clustering_cqat(cqat_model)

  # Compare between clustering and cqat in terms of FP32 accuracy
  # and numbers of unique weights
  print('FP32 accuracy of clustered model:', clst_acc)
  print_unique_weights(clustered_model_stripped)
  print('FP32 accuracy of cqat model:', cqat_acc)
  print_unique_weights(cqat_model_stripped)

  # See consistency of accuracy from TF to TFLite
  converter = tf.lite.TFLiteConverter.from_keras_model(cqat_model_stripped)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  cqat_model_stripped_tflite = converter.convert()
  interpreter = tf.lite.Interpreter(model_content=cqat_model_stripped_tflite)
  interpreter.allocate_tensors()
  test_accuracy = evaluate_model(interpreter, test_images, test_labels)

  with open('cqat.tflite', 'wb') as f:
    f.write(cqat_model_stripped_tflite)

  print('CQAT TFLite test_accuracy:', test_accuracy)
  print('CQAT TF test accuracy:', cqat_acc)


if __name__ == '__main__':
  absl_app.run(main)
