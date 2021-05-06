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
"""Tests for tf.keras pruning tools in sparsity_tooling.py."""

import tensorflow as tf

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.python.core.sparsity.keras import test_utils
from tensorflow_model_optimization.python.core.sparsity.keras.tools import sparsity_tooling

keras = tf.keras
test = tf.test


class SparsityToolingTest(test.TestCase):

  def test_prune_model(self):
    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(10,)),
        keras.layers.Dense(2),
    ])
    pruned_model = sparsity_tooling.prune_for_benchmark(
        model, target_sparsity=.8, block_size=(1, 1))

    for layer in pruned_model.layers:
      self.assertEqual((1, 1), layer.block_size)
    test_utils.assert_model_sparsity(self, 0.8, pruned_model)

  def test_prune_model_recursively(self):
    """Checks that models are recursively pruned."""

    # Setup a model with one layer being a keras.Model.
    internal_model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(10,)),
    ])
    model = keras.Sequential([
        internal_model,
        keras.layers.Dense(20),
    ])
    pruned_model = sparsity_tooling.prune_for_benchmark(
        model, target_sparsity=.8, block_size=(1, 1))

    test_utils.assert_model_sparsity(self, 0.8, pruned_model)

    # Check the block size of the prunned layers
    prunned_dense_layers = [
        layer
        for layer in pruned_model.submodules
        if isinstance(layer, pruning_wrapper.PruneLowMagnitude)
    ]
    self.assertEqual(2, len(prunned_dense_layers))
    for layer in prunned_dense_layers:
      self.assertEqual((1, 1), layer.block_size)


if __name__ == '__main__':
  test.main()
