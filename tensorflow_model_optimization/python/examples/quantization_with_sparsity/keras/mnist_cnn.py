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
"""Prune preserve Quant-Aware Training(pqat) with simple convnet on the MNIST dataset.

As a experimental feature, only `quantize_apply` been enabled with boolean flag
`prune_preserve`
"""
from __future__ import print_function

from absl import app as absl_app
import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras.collaborative_optimizations.prune_preserve import (
    default_8bit_prune_preserve_quantize_scheme,)
from tensorflow_model_optimization.python.core.quantization.keras.collaborative_optimizations.prune_preserve import (
    prune_preserve_callbacks,)
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule


layers = tf.keras.layers


def build_sequential_model(input_shape=(28, 28)):
  num_classes = 12

  return tf.keras.Sequential([
      layers.InputLayer(input_shape=input_shape),
      layers.Conv2D(32,
                    5,
                    padding='same',
                    activation='relu',
                    input_shape=input_shape),
      layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
      layers.Conv2D(64, 5, padding='same', activation='relu'),
      layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
      layers.Flatten(),
      layers.Dense(1024, activation='relu'),
      layers.Dropout(0.4),
      layers.Dense(num_classes, activation='softmax')
  ])


def compile_and_fit(model,
                    image_train,
                    label_train,
                    compile_kwargs,
                    fit_kwargs):
  # Compile the model.
  compile_args = {
      'optimizer': 'adam',
      'loss': 'sparse_categorical_crossentropy',
      'metrics': ['accuracy'],
  }
  compile_args.update(compile_kwargs)
  model.compile(**compile_args)

  # train the model.
  fit_args = {'epochs': 4, 'validation_split': 0.1}
  fit_args.update(fit_kwargs)
  model.fit(image_train, label_train, **fit_args)


def evaluate_and_show_sparsity(model, image_test, label_test):
  score = model.evaluate(image_test, label_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

  for layer in model.layers:
    if isinstance(layer,
                  prune.pruning_wrapper.PruneLowMagnitude) or isinstance(
                      layer, quantize.quantize_wrapper.QuantizeWrapper):
      for weights in layer.trainable_weights:
        np_weights = tf.keras.backend.get_value(weights)
        sparsity = 1.0 - np.count_nonzero(np_weights) / float(np_weights.size)
        print(layer.layer.__class__.__name__, ' (', weights.name,
              ') sparsity: ', sparsity)


def prune_model(original_model, train_images, train_labels):
  batch_size = 256
  epochs = 5

  pruning_params = {
      'pruning_schedule':
      pruning_schedule.ConstantSparsity(0.75, begin_step=0, frequency=100)
  }
  pruning_model = prune.prune_low_magnitude(original_model, **pruning_params)
  pruning_model.summary()

  callbacks = [pruning_callbacks.UpdatePruningStep()]
  fit_kwargs = {
      'batch_size': batch_size,
      'epochs': epochs,
      'callbacks': callbacks,
  }
  compile_and_fit(pruning_model,
                  train_images,
                  train_labels,
                  compile_kwargs={},
                  fit_kwargs=fit_kwargs)

  return pruning_model


def prune_preserve_quantize_model(pruned_model, train_images, train_labels):
  batch_size = 256
  epochs = 5

  pruned_model = prune.strip_pruning(pruned_model)
  # Prune preserve QAT model
  quant_aware_annotate_model = quantize.quantize_annotate_model(pruned_model)
  quant_aware_model = quantize.quantize_apply(
      quant_aware_annotate_model,
      scheme=default_8bit_prune_preserve_quantize_scheme
      .Default8BitPrunePreserveQuantizeScheme())
  quant_aware_model.summary()

  callbacks = [prune_preserve_callbacks.PrunePreserve()]
  fit_kwargs = {
      'batch_size': batch_size,
      'epochs': epochs,
      'callbacks': callbacks,
  }
  compile_and_fit(quant_aware_model,
                  train_images,
                  train_labels,
                  compile_kwargs={},
                  fit_kwargs=fit_kwargs)

  return quant_aware_model


def main(unused_args):
  # Load the MNIST dataset.
  mnist = tf.keras.datasets.mnist
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
  # data preprocessing
  # normalize the input images so that each pixel value is between 0 and 1.
  train_images = train_images / 255.0
  test_images = test_images / 255.0
  train_images = tf.expand_dims(train_images, axis=-1)
  test_images = tf.expand_dims(test_images, axis=-1)
  input_shape = train_images.shape[1:]
  print('train_images shape:', train_images.shape)
  print(train_images.shape[0], 'train samples')
  print(test_images.shape[0], 'test samples')

  model = build_sequential_model(input_shape)

  pruned_model = prune_model(model, train_images, train_labels)
  evaluate_and_show_sparsity(pruned_model, test_images, test_labels)

  pqat_model = prune_preserve_quantize_model(pruned_model, train_images,
                                             train_labels)
  evaluate_and_show_sparsity(pqat_model, test_images, test_labels)


if __name__ == '__main__':
  absl_app.run(main)
