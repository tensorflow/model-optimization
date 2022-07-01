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


def get_weight_size_in_bytes(weight):
  if weight.dtype == tf.string:
    return tf.reduce_sum(tf.strings.length(weight, unit="BYTE"))
  else:
    return tf.size(weight) * weight.dtype.size


def zip_directory(dir_name):
  return shutil.make_archive(dir_name, "zip", dir_name)


class EPRTest(parameterized.TestCase, tf.test.TestCase):

  def get_algorithm(self, regularization_weight=1.):
    return epr.EPR(regularization_weight=regularization_weight)

  def save_model(self, model):
    model_dir = self.create_tempdir().full_path
    model.save(model_dir)
    return model_dir

  @parameterized.parameters([5], [2, 3], [3, 4, 2], [2, 3, 4, 1])
  def test_project_training_weights_has_gradients(self, *shape):
    algorithm = self.get_algorithm()
    init = tf.ones(shape, dtype=tf.float32)
    algorithm.init_training_weights(init)
    layer = tf.keras.layers.Layer()
    for weight_repr in algorithm.weight_reprs:
      layer.add_weight(*weight_repr.args, **weight_repr.kwargs)
    with tf.GradientTape() as tape:
      weight = algorithm.project_training_weights(*layer.weights)
    gradients = tape.gradient(weight, layer.weights)
    self.assertAllEqual(
        [g is not None for g in gradients],
        [w.dtype.is_floating and "log_scale" not in w.name
         for w in layer.weights])

  @parameterized.parameters([5], [2, 3], [3, 4, 2], [2, 3, 4, 1])
  def test_regularization_loss_has_gradients(self, *shape):
    algorithm = self.get_algorithm()
    init = tf.ones(shape, dtype=tf.float32)
    algorithm.init_training_weights(init)
    layer = tf.keras.layers.Layer()
    for weight_repr in algorithm.weight_reprs:
      layer.add_weight(*weight_repr.args, **weight_repr.kwargs)
    with tf.GradientTape() as tape:
      loss = algorithm.regularization_loss(*layer.weights)
    gradients = tape.gradient(loss, layer.weights)
    self.assertAllEqual(
        [g is not None for g in gradients],
        [w.dtype.is_floating for w in layer.weights])

  @parameterized.parameters(
      ((2, 3), tf.keras.layers.Dense, 5),
      # TODO(jballe): This fails with: 'You called `set_weights(weights)` on
      # layer "private__training_wrapper" with a weight list of length 0, but
      # the layer was expecting 5 weights.' Find fix.
      # ((3, 10, 2), tf.keras.layers.Conv1D, 5, 3),
      ((1, 8, 9, 2), tf.keras.layers.Conv2D, 5, 3),
  )
  def test_model_has_gradients(self, input_shape, layer_cls, *args):
    algorithm = self.get_algorithm()
    model = tf.keras.Sequential([layer_cls(*args, use_bias=True)])
    inputs = tf.random.normal(input_shape)
    model(inputs)
    training_model = algorithm.get_training_model(model)
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(inputs)
      outputs = training_model(inputs)
      loss = tf.reduce_sum(abs(outputs)) + tf.reduce_sum(training_model.losses)
    self.assertIsNotNone(tape.gradient(loss, inputs))
    gradients = tape.gradient(loss, training_model.trainable_weights)
    self.assertAllEqual(
        [g is not None for g in gradients],
        [w.dtype.is_floating for w in training_model.trainable_weights])

  @parameterized.parameters([5], [2, 3], [3, 4, 2], [2, 3, 4, 1])
  def test_train_and_test_weights_are_equal(self, *shape):
    algorithm = self.get_algorithm()
    init = tf.random.uniform(shape, dtype=tf.float32)
    algorithm.init_training_weights(init)
    layer = tf.keras.layers.Layer()
    for weight_repr in algorithm.weight_reprs:
      layer.add_weight(*weight_repr.args, **weight_repr.kwargs)
    train_weight = algorithm.project_training_weights(*layer.weights)
    compressed_weights = algorithm.compress_training_weights(*layer.weights)
    test_weight = algorithm.decompress_weights(*compressed_weights)
    self.assertAllEqual(train_weight, test_weight)

  @parameterized.parameters([5], [2, 3], [3, 4, 2], [2, 3, 4, 1])
  def test_initialized_value_is_close_enough(self, *shape):
    algorithm = self.get_algorithm()
    init = tf.random.uniform(shape, -10., 10., dtype=tf.float32)
    algorithm.init_training_weights(init)
    layer = tf.keras.layers.Layer()
    for weight_repr in algorithm.weight_reprs:
      layer.add_weight(*weight_repr.args, **weight_repr.kwargs)
    weight = algorithm.project_training_weights(*layer.weights)
    quantization_noise_std_dev = tf.exp(-4.) / tf.sqrt(12.)
    self.assertLess(
        tf.sqrt(tf.reduce_mean(tf.square(init - weight))),
        3. * quantization_noise_std_dev)

  def test_reduces_model_size_at_reasonable_accuracy(self):
    algorithm = self.get_algorithm()
    model = build_model()
    training_model = algorithm.get_training_model(model)
    train_model(training_model)
    compressed_model = algorithm.compress_model(training_model)
    original_model_dir = self.save_model(model)
    compressed_model_dir = self.save_model(compressed_model)

    with self.subTest("training_model_has_reasonable_accuracy"):
      accuracy = evaluate_model(training_model)
      self.assertGreater(accuracy, .9)

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

    with self.subTest("compressed_model_has_reasonable_accuracy"):
      compressed_model = tf.keras.models.load_model(compressed_model_dir)
      accuracy = evaluate_model(compressed_model)
      self.assertGreater(accuracy, .9)

  def test_unregularized_training_model_has_reasonable_accuracy(self):
    algorithm = self.get_algorithm(regularization_weight=0.)
    model = build_model()
    training_model = algorithm.get_training_model(model)
    train_model(training_model)
    accuracy = evaluate_model(training_model)
    self.assertGreater(accuracy, .9)


class FastEPRTest(EPRTest):

  def get_algorithm(self, regularization_weight=1.):
    return epr.FastEPR(regularization_weight=regularization_weight, alpha=1e-2)


if __name__ == "__main__":
  tf.test.main()
