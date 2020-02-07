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
"""Integration test which ensures user facing code paths work."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

# TODO(b/139939526): move to public API.
from tensorflow.python.keras import keras_parameterized

from tensorflow_model_optimization.python.core.keras import compat
from tensorflow_model_optimization.python.core.keras import test_utils
from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras import utils


# TODO(tfmot): enable for v1. Currently fails because the decorator
# on graph mode wraps everything in a graph, which is not compatible
# with the TFLite converter's call to clear_session().
@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class QuantizeIntegrationTest(tf.test.TestCase, parameterized.TestCase):

  @staticmethod
  def _batch(dims, batch_size):
    """Adds provided batch_size to existing dims.

    If dims is (None, 5, 2), returns (batch_size, 5, 2)

    Args:
      dims: Dimensions
      batch_size: batch_size

    Returns:
      dims with batch_size added as first parameter of list.
    """
    if dims[0] is None:
      dims[0] = batch_size
    return dims

  def _assert_models_equal(self, model1, model2):
    model1_config = model1.get_config()
    model1_config.pop('build_input_shape', None)
    model2_config = model2.get_config()
    model2_config.pop('build_input_shape', None)
    self.assertEqual(model1_config, model2_config)
    self.assertAllClose(model1.get_weights(), model2.get_weights())

    inputs = np.random.randn(
        *self._batch(model1.input.get_shape().as_list(), 1))
    self.assertAllClose(model1.predict(inputs), model2.predict(inputs))

  # After saving a model to SavedModel and then loading it back,
  # the class changes, which results in config differences. This
  # may change after a sync (TF 2.2.0): TODO(alanchiao): try it.
  def _assert_saved_models_equal(self, model1, model2):
    inputs = np.random.randn(
        *self._batch(model1.input.get_shape().as_list(), 1))
    self.assertAllClose(model1.predict(inputs), model2.predict(inputs))

  # TODO(tfmot): use shared test util that is model-independent.
  @staticmethod
  def _train_model(model):
    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(
        np.random.rand(20, 10),
        tf.keras.utils.to_categorical(np.random.randint(5, size=(20, 1)), 5),
        batch_size=20)

  # TODO(pulkitb): Parameterize and add more model/runtime options.
  def testSerialization_KerasModel(self):
    model = test_utils.build_simple_dense_model()
    quantized_model = quantize.quantize(model)
    self._train_model(quantized_model)

    _, model_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(quantized_model, model_file)
    with quantize.quantize_scope():
      loaded_model = tf.keras.models.load_model(model_file)

    self._assert_models_equal(quantized_model, loaded_model)

  def testSerialization_TF2SavedModel(self):
    if compat.is_v1_apis():
      return

    model = test_utils.build_simple_dense_model()
    quantized_model = quantize.quantize(model)
    self._train_model(quantized_model)

    model_dir = tempfile.mkdtemp()
    tf.keras.models.save_model(quantized_model, model_dir)
    loaded_model = tf.keras.models.load_model(model_dir)

    self._assert_saved_models_equal(quantized_model, loaded_model)


if __name__ == '__main__':
  tf.test.main()
