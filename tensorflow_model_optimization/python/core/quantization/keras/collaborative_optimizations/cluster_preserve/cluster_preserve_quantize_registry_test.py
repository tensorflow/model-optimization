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
"""Tests for ClusterPreserveQuantizeRegistry."""

from absl.testing import parameterized

import tensorflow as tf

from tensorflow.python.keras import keras_parameterized

from tensorflow_model_optimization.python.core.quantization.keras import quantize_config
from tensorflow_model_optimization.python.core.clustering.keras import clustering_registry
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.collaborative_optimizations.cluster_preserve import cluster_preserve_quantize_registry

QuantizeConfig = quantize_config.QuantizeConfig
layers = tf.keras.layers


@keras_parameterized.run_all_keras_modes
class ClusterPreserveQuantizeRegistryTest(tf.test.TestCase,
                                          parameterized.TestCase):
  def setUp(self):
    super(ClusterPreserveQuantizeRegistryTest, self).setUp()
    self.cluster_preserve_quantize_registry = (
        cluster_preserve_quantize_registry.ClusterPreserveQuantizeRegistry())
    # layers which are supported
    # initial and build a Conv2D layer
    self.layer_conv2d = layers.Conv2D(10, (2, 2))
    self.layer_conv2d.build((2, 2))
    # initial and build a Dense layer
    self.layer_dense = layers.Dense(10)
    self.layer_dense.build((2, 2))
    # initial and build a ReLU layer
    self.layer_relu = layers.ReLU()
    self.layer_relu.build((2, 2))

    # a layer which is not supported
    # initial and build a Custom layer
    self.layer_custom = self.CustomLayer()
    self.layer_custom.build()

  class CustomLayer(layers.Layer):
    # a simple custom layer with training weights
    def __init__(self):
      super(ClusterPreserveQuantizeRegistryTest.CustomLayer, self).__init__()

    def build(self, input_shape=(2, 2)):
      self.add_weight(shape=input_shape,
                      initializer="random_normal",
                      trainable=True)

  class CustomQuantizeConfig(QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
      return []

    def get_activations_and_quantizers(self, layer):
      return []

    def set_quantize_weights(self, layer, quantize_weights):
      pass

    def set_quantize_activations(self, layer, quantize_activations):
      pass

    def get_output_quantizers(self, layer):
      return []

    def get_config(self):
      return {}

  def testSupportsKerasLayer(self):
    # test registered layer
    self.assertTrue(
        self.cluster_preserve_quantize_registry.supports(self.layer_dense))
    self.assertTrue(
        self.cluster_preserve_quantize_registry.supports(self.layer_conv2d))
    # test layer without training weights
    self.assertTrue(
        self.cluster_preserve_quantize_registry.supports(self.layer_relu))

  def testDoesNotSupportCustomLayer(self):
    self.assertFalse(
        self.cluster_preserve_quantize_registry.supports(self.layer_custom))

  def testApplyClusterPreserveWithQuantizeConfig(self):
    self.cluster_preserve_quantize_registry.\
      apply_cluster_preserve_quantize_config(
        self.layer_conv2d,
        default_8bit_quantize_registry.Default8BitConvQuantizeConfig(
            ['kernel'], ['activation'], False))

  def testRaisesErrorUnsupportedQuantizeConfigWithLayer(self):
    with self.assertRaises(
      ValueError, msg="Unregistered QuantizeConfigs should raise error."):
      self.cluster_preserve_quantize_registry.\
        apply_cluster_preserve_quantize_config(
          self.layer_conv2d, self.CustomQuantizeConfig)

    with self.assertRaises(ValueError,
                           msg="Unregistered layers should raise error."):
      self.cluster_preserve_quantize_registry.\
        apply_cluster_preserve_quantize_config(
          self.layer_custom, self.CustomQuantizeConfig)


class ClusterPreserveDefault8bitQuantizeRegistryTest(tf.test.TestCase):
  def setUp(self):
    super(ClusterPreserveDefault8bitQuantizeRegistryTest, self).setUp()
    self.default_8bit_quantize_registry = (
        default_8bit_quantize_registry.Default8BitQuantizeRegistry())
    self.cluster_registry = clustering_registry.ClusteringRegistry()
    self.cluster_preserve_quantize_registry = (
        cluster_preserve_quantize_registry.ClusterPreserveQuantizeRegistry())

  def testSupportsClusterDefault8bitQuantizeKerasLayers(self):
    # ClusterPreserveQuantize supported layer, must be suppoted
    # by both Cluster and Quantize
    cqat_layers_config_map = \
      self.cluster_preserve_quantize_registry._LAYERS_CONFIG_MAP
    for cqat_support_layer in cqat_layers_config_map:
      if cqat_layers_config_map[cqat_support_layer].weight_attrs and \
        cqat_layers_config_map[cqat_support_layer].quantize_config_attrs:
        self.assertTrue(
            cqat_support_layer in self.cluster_registry._LAYERS_WEIGHTS_MAP,
            msg="Clusteirng doesn't support {}".format(cqat_support_layer))
        self.assertTrue(
            cqat_support_layer
            in self.default_8bit_quantize_registry._layer_quantize_map,
            msg="Default 8bit QAT doesn't support {}".format(
                cqat_support_layer))


if __name__ == '__main__':
  tf.test.main()
