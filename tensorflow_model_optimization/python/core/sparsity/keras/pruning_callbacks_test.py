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

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import backend as K
from tensorflow.python.platform import test
from tensorflow_model_optimization.python.core.keras import test_utils as keras_test_utils
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks


@tf_test_util.run_all_in_graph_and_eager_modes
class PruneTest(test.TestCase):

  def testUpdatesPruningStep(self):
    model = prune.prune_low_magnitude(
        keras_test_utils.build_simple_dense_model())
    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(
        np.random.rand(20, 10),
        keras.utils.np_utils.to_categorical(
            np.random.randint(5, size=(20, 1)), 5),
        batch_size=20,
        epochs=3,
        callbacks=[pruning_callbacks.UpdatePruningStep()])

    self.assertEqual(2, K.get_value(model.layers[0].pruning_step))
    self.assertEqual(2, K.get_value(model.layers[1].pruning_step))


if __name__ == '__main__':
  test.main()
