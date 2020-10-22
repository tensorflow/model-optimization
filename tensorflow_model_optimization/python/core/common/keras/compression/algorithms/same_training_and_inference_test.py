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
"""Tests for when the training and inference graphs are the same."""

import os
import tempfile

import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.common.keras.compression.algorithms import same_training_and_inference as svd


def _build_model():
  i = tf.keras.layers.Input(shape=(28, 28), name='input')
  x = tf.keras.layers.Reshape((28, 28, 1))(i)
  x = tf.keras.layers.Conv2D(
      20, 5, activation='relu', padding='valid', name='conv1')(
          x)
  x = tf.keras.layers.MaxPool2D(2, 2)(x)
  x = tf.keras.layers.Conv2D(
      50, 5, activation='relu', padding='valid', name='conv2')(
          x)
  x = tf.keras.layers.MaxPool2D(2, 2)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(500, activation='relu', name='fc1')(x)
  output = tf.keras.layers.Dense(10, name='fc2')(x)

  model = tf.keras.Model(inputs=[i], outputs=[output])
  return model


def _get_dataset():
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  # Use subset of 60000 examples to keep unit test speed fast.
  x_train = x_train[0:1000]
  y_train = y_train[0:1000]
  return (x_train, y_train), (x_test, y_test)


def _train_model(model):
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

  (x_train, y_train), _ = _get_dataset()

  model.fit(x_train, y_train, epochs=1)


def _save_as_saved_model(model):
  saved_model_dir = tempfile.mkdtemp()
  model.save(saved_model_dir)
  return saved_model_dir


# TODO(tfmot): reuse existing test utilities.
def _convert_to_tflite(saved_model_dir):
  _, tflite_file = tempfile.mkstemp()

  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  tflite_model = converter.convert()

  with open(tflite_file, 'wb') as f:
    f.write(tflite_model)

  return tflite_file


# TODO(tfmot): reuse test_utils_mnist.py.
def _test_tflite(tflite_file):
  interpreter = tf.lite.Interpreter(model_path=tflite_file)
  interpreter.allocate_tensors()

  input_index = interpreter.get_input_details()[0]['index']
  output_index = interpreter.get_output_details()[0]['index']

  (_, _), (x_test, y_test) = _get_dataset()

  # Testing the entire dataset is too slow. Verifying only 300 of 10k samples.
  x_test = x_test[0:300, :]
  y_test = y_test[0:300]

  total_seen = 0
  num_correct = 0

  for img, label in zip(x_test, y_test):
    batch_input_shape = (1, 28, 28)
    inp = img.reshape(batch_input_shape)
    inp = inp.astype(np.float32)
    total_seen += 1
    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)

    if np.argmax(predictions) == label:
      num_correct += 1

  return float(num_correct) / float(total_seen)


def _get_directory_size_in_bytes(directory):
  total = 0
  try:
    for entry in os.scandir(directory):
      if entry.is_file():
        # if it's a file, use stat() function
        total += entry.stat().st_size
      elif entry.is_dir():
        # if it's a directory, recursively call this function
        total += _get_directory_size_in_bytes(entry.path)
  except NotADirectoryError:
    # if `directory` isn't a directory, get the file size then
    return os.path.getsize(directory)
  except PermissionError:
    # if for whatever reason we can't open the folder, return 0
    return 0
  return total


class FunctionalTest(tf.test.TestCase):

  # TODO(tfmot): can simplify to single layer test that checks exact
  # dimensions of weights.
  def testSVD_ReducesSavedModelSize(self):
    model = _build_model()

    original_saved_model_dir = _save_as_saved_model(model)

    params = svd.SVDParams(rank=16)
    compressed_model = svd.optimize(model, params)

    saved_model_dir = _save_as_saved_model(compressed_model)

    original_size = _get_directory_size_in_bytes(original_saved_model_dir)
    compressed_size = _get_directory_size_in_bytes(saved_model_dir)

    self.assertLess(compressed_size, original_size / 3)

  def testSVD_HasReasonableAccuracy_TF(self):
    model = _build_model()

    params = svd.SVDParams(rank=16)
    compressed_model = svd.optimize(model, params)

    _train_model(compressed_model)

    _, (x_test, y_test) = _get_dataset()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    compressed_model.compile(
        optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    results = compressed_model.evaluate(x_test, y_test)

    self.assertGreater(results[1], 0.60)

  def testSVD_ReducesTFLiteModelSize(self):
    model = _build_model()

    original_saved_model_dir = _save_as_saved_model(model)
    original_tflite_file = _convert_to_tflite(original_saved_model_dir)

    params = svd.SVDParams(rank=16)
    compressed_model = svd.optimize(model, params)

    saved_model_dir = _save_as_saved_model(compressed_model)
    compressed_tflite_file = _convert_to_tflite(saved_model_dir)

    original_size = os.path.getsize(original_tflite_file)
    compressed_size = os.path.getsize(compressed_tflite_file)

    self.assertLess(compressed_size, original_size / 6)

  def testSVD_HasReasonableAccuracy_TFLite(self):
    model = _build_model()

    params = svd.SVDParams(rank=16)
    compressed_model = svd.optimize(model, params)

    _train_model(compressed_model)

    saved_model_dir = _save_as_saved_model(compressed_model)
    compressed_tflite_file = _convert_to_tflite(saved_model_dir)

    accuracy = _test_tflite(compressed_tflite_file)

    self.assertGreater(accuracy, 0.60)

  # TODO(tfmot): can simplify to single layer test.
  def testSVD_BreaksDownLayerWeights(self):
    model = _build_model()

    first_conv_layer = model.layers[2]
    self.assertLen(first_conv_layer.weights, 2)

    params = svd.SVDParams(rank=16)
    compressed_model = svd.optimize(model, params)

    first_conv_layer = compressed_model.layers[2]

    self.assertLen(first_conv_layer.weights, 3)

  # TODO(tfmot): can simplify to single layer test.
  def testSVD_PreservesPretrainedWeights(self):
    i = tf.keras.layers.Input(shape=(2), name='input')
    output = tf.keras.layers.Dense(3, name='fc1')(i)
    model = tf.keras.Model(inputs=[i], outputs=[output])

    dense_layer_weights = model.layers[1].get_weights()

    params = svd.SVDParams(rank=1)
    compressed_model = svd.optimize(model, params)

    dense_layer_compressed_weights = compressed_model.layers[1].get_weights()

    # kernel
    algorithm = svd.SVD(params)
    w1_repr, w2_repr = algorithm.init_training_weights_repr(
        dense_layer_weights[0])
    assert ((w1_repr.initializer(None) == dense_layer_compressed_weights[0]
            ).numpy().all())
    assert ((w2_repr.initializer(None) == dense_layer_compressed_weights[1]
            ).numpy().all())

    # bias
    assert (dense_layer_weights[1] == dense_layer_compressed_weights[2]).all()


if __name__ == '__main__':
  tf.test.main()
