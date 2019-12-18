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
"""Tests for Pruning callbacks."""

import os
import tempfile

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.keras import test_utils as keras_test_utils
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks

# TODO(b/139939526): move to public API.


@keras_parameterized.run_all_keras_modes
class PruneTest(tf.test.TestCase, parameterized.TestCase):

  def _assertLogsExist(self, log_dir):
    self.assertNotEmpty(os.listdir(log_dir))

  def testUpdatePruningStepsAndLogsSummaries(self):
    log_dir = tempfile.mkdtemp()
    model = prune.prune_low_magnitude(
        keras_test_utils.build_simple_dense_model())
    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(
        np.random.rand(20, 10),
        tf.keras.utils.to_categorical(np.random.randint(5, size=(20, 1)), 5),
        batch_size=20,
        epochs=3,
        callbacks=[
            pruning_callbacks.UpdatePruningStep(),
            pruning_callbacks.PruningSummaries(log_dir=log_dir)
        ])

    self.assertEqual(2,
                     tf.keras.backend.get_value(model.layers[0].pruning_step))
    self.assertEqual(2,
                     tf.keras.backend.get_value(model.layers[1].pruning_step))

    self._assertLogsExist(log_dir)

if __name__ == '__main__':
  tf.test.main()
