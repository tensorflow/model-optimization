# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for EPR algorithm."""

import os
import shutil
from absl.testing import parameterized
import tensorflow as tf
from tensorflow_model_optimization.python.core.common.keras.compression.algorithms import epr


def build_model():
  inputs = tf.keras.layers.Input(shape=(28, 28), name="input")
  x = tf.keras.layers.Reshape((28, 28, 1))(inputs)
  x = tf.keras.layers.Conv2D(
      20, 5, use_bias=True, activation="relu", padding="valid", name="conv1")(x)
  x = tf.keras.layers.MaxPool2D(2, 2)(x)
  x = tf.keras.layers.Conv2D(
      50, 5, use_bias=True, activation="relu", padding="valid", name="conv2")(x)
  x = tf.keras.layers.MaxPool2D(2, 2)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(
      500, use_bias=True, activation="relu", name="fc1")(x)
  outputs = tf.keras.layers.Dense(
      10, use_bias=True, name="fc2")(x)
  return tf.keras.Model(inputs=[inputs], outputs=[outputs])


def get_dataset():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_train = (x_train / 255).astype("float32")
  x_test = (x_test / 255).astype("float32")
  return (x_train, y_train), (x_test, y_test)


def train_model(model):
  model.compile(
      optimizer=tf.keras.optimizers.Adam(1e-2),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
  )
  (x_train, y_train), _ = get_dataset()
  model.fit(x_train, y_train, batch_size=128, epochs=3)


def evaluate_model(model):
  model.compile(
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
  )
  _, (x_test, y_test) = get_dataset()
  results = model.evaluate(x_test, y_test, batch_size=128, return_dict=True)
  return results["accuracy"]


def train_and_compress_model():
  model = build_model()
  algorithm = epr.EPR(entropy_penalty=10.)
  training_model = algorithm.get_training_model(model)
  train_model(training_model)
  compressed_model = algorithm.compress_model(training_model)
  return model, training_model, compressed_model


def get_weight_size_in_bytes(weight):
  if weight.dtype == tf.string:
    return tf.reduce_sum(tf.strings.length(weight, unit="BYTE"))
  else:
    return tf.size(weight) * weight.dtype.size


def zip_directory(dir_name):
  return shutil.make_archive(dir_name, "zip", dir_name)


class EPRTest(parameterized.TestCase, tf.test.TestCase):

  def _save_models(self, model, compressed_model):
    model_dir = self.create_tempdir().full_path
    original_model_dir = os.path.join(model_dir, "original")
    compressed_model_dir = os.path.join(model_dir, "compressed")
    model.save(original_model_dir)
    compressed_model.save(compressed_model_dir)
    return original_model_dir, compressed_model_dir

  @parameterized.parameters([5], [2, 3], [3, 4, 2], [2, 3, 4, 1])
  def test_project_training_weights_has_gradients(self, *shape):
    algorithm = epr.EPR(entropy_penalty=1.)
    init = tf.ones(shape, dtype=tf.float32)
    algorithm.init_training_weights(init)
    layer = tf.keras.layers.Layer()
    for weight_repr in algorithm.weight_reprs:
      layer.add_weight(*weight_repr.args, **weight_repr.kwargs)
    with tf.GradientTape() as tape:
      weight = algorithm.project_training_weights(*layer.weights)
    gradients = tape.gradient(weight, layer.weights)
    # Last weight is scale of prior. Should not have a gradient here.
    self.assertAllEqual(
        [g is not None for g in gradients],
        [w.dtype.is_floating for w in layer.weights[:-1]] + [False])

  @parameterized.parameters([5], [2, 3], [3, 4, 2], [2, 3, 4, 1])
  def test_compute_entropy_has_gradients(self, *shape):
    algorithm = epr.EPR(entropy_penalty=1.)
    init = tf.ones(shape, dtype=tf.float32)
    algorithm.init_training_weights(init)
    layer = tf.keras.layers.Layer()
    for weight_repr in algorithm.weight_reprs:
      layer.add_weight(*weight_repr.args, **weight_repr.kwargs)
    with tf.GradientTape() as tape:
      loss = algorithm.compute_entropy(*layer.weights)
    gradients = tape.gradient(loss, layer.weights)
    self.assertAllEqual(
        [g is not None for g in gradients],
        [w.dtype.is_floating for w in layer.weights])

  @parameterized.parameters([5], [2, 3], [3, 4, 2], [2, 3, 4, 1])
  def test_train_and_test_weights_are_equal(self, *shape):
    algorithm = epr.EPR(entropy_penalty=1.)
    init = tf.random.uniform(shape, dtype=tf.float32)
    algorithm.init_training_weights(init)
    layer = tf.keras.layers.Layer()
    for weight_repr in algorithm.weight_reprs:
      layer.add_weight(*weight_repr.args, **weight_repr.kwargs)
    train_weight = algorithm.project_training_weights(*layer.weights)
    compressed_weights = algorithm.compress_training_weights(*layer.weights)
    test_weight = algorithm.decompress_weights(*compressed_weights)
    self.assertAllEqual(train_weight, test_weight)

  def test_reduces_model_size_at_reasonable_accuracy(self):
    model, _, compressed_model = train_and_compress_model()
    original_model_dir, compressed_model_dir = self._save_models(
        model, compressed_model)

    with self.subTest("compressed_weights_are_smaller"):
      original_size = sum(
          map(get_weight_size_in_bytes, model.weights)).numpy()
      compressed_size = sum(
          map(get_weight_size_in_bytes, compressed_model.weights)).numpy()
      self.assertLess(compressed_size, 0.01 * original_size)

    with self.subTest("zip_compressed_model_is_smaller"):
      original_zipfile = zip_directory(original_model_dir)
      compressed_zipfile = zip_directory(compressed_model_dir)
      original_size = os.path.getsize(original_zipfile)
      compressed_size = os.path.getsize(compressed_zipfile)
      # TODO(jballe): There is a lot of overhead in the saved function graphs
      # (saved_model.pb), which is especially severe for small models like this
      # one. The function graph is several times larger than the weights. Can we
      # save some space by only saving the function graph of the main call(),
      # rather than of each layer?
      self.assertLess(compressed_size, 0.2 * original_size)

    with self.subTest("has_reasonable_accuracy"):
      compressed_model = tf.keras.models.load_model(compressed_model_dir)
      accuracy = evaluate_model(compressed_model)
      self.assertGreater(accuracy, .9)


if __name__ == "__main__":
  tf.test.main()
