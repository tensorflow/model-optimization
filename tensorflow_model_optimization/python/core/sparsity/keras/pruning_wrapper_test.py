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
"""Tests for keras pruning wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

keras = tf.keras
layers = keras.layers
Prune = pruning_wrapper.PruneLowMagnitude


class CustomLayer(keras.layers.Layer):
  """A custom layer which is not prunable."""

  def __init__(self, input_dim=16, output_dim=32):
    super(CustomLayer, self).__init__()
    self.weight = self.add_weight(
        shape=(input_dim, output_dim),
        initializer='random_normal',
        trainable=True)
    self.bias = self.add_weight(
        shape=(output_dim),
        initializer='zeros',
        trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.weight) + self.bias


class CustomLayerPrunable(CustomLayer):
  """A prunable custom layer.

  The layer is same with the CustomLayer except it has a 'get_prunable_weights'
  attribute.
  """

  def get_prunable_weights(self):
    return [self.weight, self.bias]


class PruningWrapperTest(tf.test.TestCase):

  def setUp(self):
    self.model = keras.Sequential()
    self.params = {
        'pruning_schedule': pruning_schedule.ConstantSparsity(0.5, 0),
        'block_size': (1, 1),
        'block_pooling_type': 'AVG'
    }

  def testPruneWrapperAllowsOnlyValidPoolingType(self):
    layer = layers.Dense(10)
    with self.assertRaises(ValueError):
      pruning_wrapper.PruneLowMagnitude(layer, block_pooling_type='MIN')

    pruning_wrapper.PruneLowMagnitude(layer, block_pooling_type='AVG')
    pruning_wrapper.PruneLowMagnitude(layer, block_pooling_type='MAX')

  def _check_mask_count(self, expected_mask_count=0):
    mask_count = 0
    for l in self.model.layers:
      mask_count += len(l.pruning_vars)
    self.assertEqual(mask_count, expected_mask_count)

  # TODO(suyoggupta): Randomize the layer dimensions
  def testDense(self):
    self.model.add(Prune(layers.Dense(10), **self.params))
    self.model.build(input_shape=(10, 1))

    self._check_mask_count(expected_mask_count=1)

  def testEmbedding(self):
    self.model.add(
        Prune(
            layers.Embedding(10, 10),
            input_shape=(10,),
            **self.params))
    self.model.build(input_shape=(10, 1))

    self._check_mask_count(expected_mask_count=1)

  def testConv2D(self):
    self.model.add(Prune(layers.Conv2D(4, (3, 3)), **self.params))
    self.model.build(input_shape=(1, 16, 16, 4))

    self._check_mask_count(expected_mask_count=1)

  def testPruneModel(self):
    self.model.add(Prune(layers.Conv2D(32, 5)))
    self.model.add(
        Prune(layers.MaxPooling2D((2, 2), (2, 2))))
    self.model.add(Prune(layers.Conv2D(64, 5)))
    self.model.add(
        Prune(layers.MaxPooling2D((2, 2), (2, 2))))
    self.model.add(Prune(layers.Flatten()))
    self.model.add(Prune(layers.Dense(1024)))
    self.model.add(Prune(layers.Dropout(0.4)))
    self.model.add(Prune(layers.Dense(10)))
    self.model.build(input_shape=(1, 28, 28, 1))

    self._check_mask_count(expected_mask_count=4)

    # Test serialization
    model_config = self.model.get_config()
    for layer in model_config['layers']:
      layer.pop('build_config', None)
    self.assertEqual(
        model_config,
        self.model.__class__.from_config(
            self.model.get_config(),
            custom_objects={
                'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude
            },
        ).get_config(),
    )

  def testCustomLayerNonPrunable(self):
    layer = CustomLayer(input_dim=16, output_dim=32)
    inputs = keras.layers.Input(shape=(16))
    _ = layer(inputs)
    with self.assertRaises(ValueError):
      pruning_wrapper.PruneLowMagnitude(layer, block_pooling_type='MAX')

  def testCustomLayerPrunable(self):
    layer = CustomLayerPrunable(input_dim=16, output_dim=32)
    inputs = keras.layers.Input(shape=(16))
    _ = layer(inputs)
    pruned_layer = pruning_wrapper.PruneLowMagnitude(
        layer, block_pooling_type='MAX'
    )
    # The name is the layer's name prefixed by the snake_case version of the
    # `PruneLowMagnitude` class's name.
    self.assertEqual(
        pruned_layer.name, 'prune_low_magnitude_custom_layer_prunable'
    )

  def testCollectPrunableLayers(self):
    lstm_layer = keras.layers.RNN(
        layers.LSTMCell(4, dropout=0.5, recurrent_dropout=0.5),
        input_shape=(None, 4))
    self.model.add(Prune(lstm_layer))
    self.model.add(Prune(layers.BatchNormalization()))
    self.model.add(Prune(layers.Flatten()))
    self.model.add(Prune(layers.Dense(10)))
    self.model.add(Prune(layers.Dropout(0.5)))

    self.model.build(input_shape=(1, 4))

    self.assertLen(pruning_wrapper.collect_prunable_layers(self.model), 5)

  def testConv3DWeightNotPrunedWithSparsityMbyN(self):
    layer = keras.layers.Conv3D(2, 3)
    inputs = keras.layers.Input(shape=(4, 28, 28, 28, 1))
    _ = layer(inputs)
    self.model.add(Prune(layer, sparsity_m_by_n=(2, 4)))

    pruned_layers = pruning_wrapper.collect_prunable_layers(self.model)

    self.assertLen(pruned_layers, 1)
    # Only rank-2 (e.g, Conv2D) or rank-4 (e.g, Dense) weight are pruned with
    # M-by-N sparsity.
    self.assertLen(pruned_layers[0].pruning_vars, 0)


if __name__ == '__main__':
  tf.test.main()
