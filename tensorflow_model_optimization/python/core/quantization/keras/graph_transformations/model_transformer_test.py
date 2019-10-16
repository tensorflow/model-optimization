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
"""Tests for Model Transformation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.platform import test

from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms

ModelTransformer = model_transformer.ModelTransformer
Transform = transforms.Transform
LayerPattern = transforms.LayerPattern
LayerNode = transforms.LayerNode


class ModelTransformerTest(test.TestCase):

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

  def _create_model_inputs(self, model):
    return np.random.randn(*self._batch(model.input.get_shape().as_list(), 1))

  def _simple_dense_model(self):
    inp = keras.layers.Input((3,))
    x = keras.layers.Dense(2)(inp)
    out = keras.layers.ReLU(6.0)(x)
    return keras.Model(inp, out)

  def _assert_config(self, expected_config, actual_config, exclude_keys=None):
    """Asserts that the two config dictionaries are equal.

    This method is used to compare keras Model and Layer configs. It provides
    the ability to exclude the keys we don't want compared.

    Args:
      expected_config: Config which we expect.
      actual_config: Actual received config.
      exclude_keys: List of keys to not check against.
    """
    expected_config = expected_config.copy()
    actual_config = actual_config.copy()

    def _remove_keys(config):
      """Removes all exclude_keys (including nested) from the dict."""
      for key in exclude_keys:
        if key in config:
          del config[key]

      for _, v in config.items():
        if isinstance(v, dict):
          _remove_keys(v)

        if isinstance(v, list):
          for item in v:
            if isinstance(item, dict):
              _remove_keys(item)

    if exclude_keys:
      _remove_keys(expected_config)
      _remove_keys(actual_config)

    self.assertDictEqual(expected_config, actual_config)

  def _assert_model_results_equal(self, model, transformed_model):
    inputs = self._create_model_inputs(model)
    self.assertAllClose(
        model.predict(inputs), transformed_model.predict(inputs))

  # Transform classes for testing.

  class ReplaceDenseLayer(transforms.Transform):
    """Replaces `Dense` layers with `MyDense`, a simple inherited layer.

    This `Transform` class replaces `Dense` layers with a class `MyDense`
    which is simply an empty inheritance of `Dense`. This makes it easy to test
    the transformation code.
    """

    class MyDense(keras.layers.Dense):
      pass

    def pattern(self):
      return LayerPattern('Dense')

    def replacement(self, match_layer):
      match_layer_config = match_layer.layer['config']
      my_dense_layer = self.MyDense(**match_layer_config)

      replace_layer = keras.layers.serialize(my_dense_layer)
      replace_layer['name'] = replace_layer['config']['name']

      return LayerNode(replace_layer, match_layer.weights, [])

    def custom_objects(self):
      return {'MyDense': self.MyDense}

  def testReplaceSingleLayerWithSingleLayer_OneOccurrence(self):
    model = self._simple_dense_model()

    transformed_model = ModelTransformer(
        model, [self.ReplaceDenseLayer()]).transform()

    self._assert_config(model.get_config(), transformed_model.get_config(),
                        ['class_name'])
    self.assertEqual('MyDense', transformed_model.layers[1].__class__.__name__)

    self._assert_model_results_equal(model, transformed_model)

  def testReplaceSingleLayerWithSingleLayer_MultipleOccurrences(self):
    inp = keras.layers.Input((3,))
    x1 = keras.layers.Dense(2)(inp)
    x2 = keras.layers.Dense(2)(inp)
    out1 = keras.layers.ReLU(6.0)(x1)
    out2 = keras.layers.ReLU(6.0)(x2)
    model = keras.Model(inp, [out1, out2])

    transformed_model = ModelTransformer(
        model, [self.ReplaceDenseLayer()]).transform()

    self._assert_config(model.get_config(), transformed_model.get_config(),
                        ['class_name'])
    self.assertEqual('MyDense', transformed_model.layers[1].__class__.__name__)
    self.assertEqual('MyDense', transformed_model.layers[2].__class__.__name__)

    self._assert_model_results_equal(model, transformed_model)

  def testReplaceSingleLayerWithSingleLayer_MatchParameters(self):
    class RemoveBiasInDense(transforms.Transform):
      """Replaces Dense layers with matching layers with `use_bias=False`."""

      def pattern(self):
        return LayerPattern('Dense', {'use_bias': True})

      def replacement(self, match_layer):
        match_layer_config = match_layer.layer['config']
        # Remove bias
        match_layer_weights = match_layer.weights
        match_layer_weights.popitem()

        match_layer_config['use_bias'] = False
        new_dense_layer = keras.layers.Dense(**match_layer_config)

        replace_layer = keras.layers.serialize(new_dense_layer)
        replace_layer['name'] = replace_layer['config']['name']

        return LayerNode(replace_layer, match_layer_weights, [])

    model = self._simple_dense_model()

    transformed_model = ModelTransformer(
        model, [RemoveBiasInDense()]).transform()

    self._assert_config(model.get_config(), transformed_model.get_config(),
                        ['use_bias'])
    self.assertFalse(transformed_model.layers[1].use_bias)

    # Should match since bias is initialized with zeros.
    self._assert_model_results_equal(model, transformed_model)

  def testReplaceSingleLayer_WithMultipleLayers(self):
    # TODO(pulkitb): Implement
    pass

  def testReplaceChainOfLayers_WithSingleLayer(self):
    class FuseReLUIntoDense(transforms.Transform):
      """Fuse ReLU into Dense layers."""

      def pattern(self):
        return LayerPattern('ReLU', inputs=[LayerPattern('Dense')])

      def replacement(self, match_layer):
        dense_layer_config = match_layer.input_layers[0].layer['config']
        dense_layer_weights = match_layer.input_layers[0].weights
        dense_layer_config['activation'] = 'relu'

        new_dense_layer = keras.layers.Dense(**dense_layer_config)

        replace_layer = keras.layers.serialize(new_dense_layer)
        replace_layer['name'] = replace_layer['config']['name']

        return LayerNode(replace_layer, dense_layer_weights, [])

    inp = keras.layers.Input((3,))
    out = keras.layers.Dense(2, activation='relu')(inp)
    model_fused = keras.Model(inp, out)

    inp = keras.layers.Input((3,))
    x = keras.layers.Dense(2)(inp)
    out = keras.layers.ReLU()(x)
    model = keras.Model(inp, out)
    model.set_weights(model_fused.get_weights())

    transformed_model = ModelTransformer(
        model, [FuseReLUIntoDense()]).transform()

    self._assert_config(
        model_fused.get_config(), transformed_model.get_config(),
        # Layers have different names in the models, but same config.
        # Consider verifying the names loosely.
        ['input_layers', 'output_layers', 'name', 'inbound_nodes'])

    self._assert_model_results_equal(model, transformed_model)
    self._assert_model_results_equal(model_fused, transformed_model)

  def testReplaceChainOfLayers_WithChainOfLayers(self):
    # TODO(pulkitb): Implement
    pass

  def testReplaceTreeOfLayers_WithSingleLayer(self):
    # TODO(pulkitb): Implement
    pass

  def testReplaceTreeOfLayers_WithTreeOfLayers(self):
    # TODO(pulkitb): Implement
    pass

  def testDoesNotMatchForever_IfReplacementEqualsMatch(self):
    class ReplaceWithSelf(Transform):

      def pattern(self):
        return LayerPattern('ReLU', inputs=[LayerPattern('Dense')])

      def replacement(self, match_layer):
        return match_layer

    model = self._simple_dense_model()

    transformed_model = ModelTransformer(
        model, [ReplaceWithSelf()]).transform()

    self._assert_config(model.get_config(), transformed_model.get_config())

  # Negative Tests
  # TODO(pulkitb): Add negative tests
  # 1. Does not replace if any layer in the pattern has multiple nodes/consumers
  # 2. Adding a single layer clone will lead to infinite loop. Fix and test.
  # 3. Handles layer being part of multiple models.


if __name__ == '__main__':
  test.main()
