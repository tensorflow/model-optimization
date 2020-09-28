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
"""Tests for PrunePreserveQuantizeRegistry."""

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized

from tensorflow_model_optimization.python.core.quantization.keras import quantize_config
from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.prune_preserve import prune_preserve_quantize_registry

QuantizeConfig = quantize_config.QuantizeConfig
layers = tf.keras.layers


@keras_parameterized.run_all_keras_modes
class PrunePreserveQuantizeRegistryTest(tf.test.TestCase,
                                        parameterized.TestCase):
  def setUp(self):
    super(PrunePreserveQuantizeRegistryTest, self).setUp()
    self.prune_preserve_quantize_registry = prune_preserve_quantize_registry.PrunePreserveQuantizeRegistry(
    )

  class CustomLayer(layers.Layer):
    pass

  class CustomQuantizeConfig(QuantizeConfig):
    pass

  def testSupports_KerasLayer(self):
    self.assertTrue(
        self.prune_preserve_quantize_registry.supports(layers.Dense(10)))
    self.assertTrue(
        self.prune_preserve_quantize_registry.supports(
            layers.Conv2D(10, (2, 2))))

  def testDoesNotSupport_CustomLayer(self):
    self.assertFalse(
        self.prune_preserve_quantize_registry.supports(self.CustomLayer()))

  def testApplyPrunePreserve_WithQuantizeConfig(self):
    self.prune_preserve_quantize_registry.apply_sparsity_preserve_quantize_config(
        layers.Conv2D(10, (2, 2)),
        default_8bit_quantize_registry.Default8BitConvQuantizeConfig(
            ['kernel'], ['activation'], False))

  def testRaisesError_Unsupported_QuantizeConfigWithLayer(self):
    with self.assertRaises(
        ValueError, msg="Unregistered QuantizeConfigs should raise error."):
      self.prune_preserve_quantize_registry.apply_sparsity_preserve_quantize_config(
          layers.Conv2D(10, (2, 2)), self.CustomQuantizeConfig)

    with self.assertRaises(ValueError,
                           msg="Unregistered layers should raise error."):
      self.prune_preserve_quantize_registry.apply_sparsity_preserve_quantize_config(
          self.CustomLayer, self.CustomQuantizeConfig)


class PrunePreserveDefault8bitQuantizeRegistryTest(tf.test.TestCase):
  def setUp(self):
    super(PrunePreserveDefault8bitQuantizeRegistryTest, self).setUp()
    self.default_8bit_quantize_registry = default_8bit_quantize_registry.QuantizeRegistry(
    )
    self.prune_registry = prune_registry.PruneRegistry()
    self.prune_preserve_quantize_registry = prune_preserve_quantize_registry.PrunePreserveQuantizeRegistry(
    )

  def testSupports_Prune_Default8bitQuantize_KerasLayers(self):
    # PrunePreserveQuantize supported layer, must be suppoted by both Prune and Quantize
    pqat_layers_config_map = self.prune_preserve_quantize_registry._LAYERS_CONFIG_MAP
    for pqat_support_layer in pqat_layers_config_map:
      if (pqat_layers_config_map[pqat_support_layer].weight_attrs and
          pqat_layers_config_map[pqat_support_layer].quantize_config_attrs):
        self.assertTrue(
            pqat_support_layer in self.prune_registry._LAYERS_WEIGHTS_MAP,
            msg="Prune doesn't support {}".format(pqat_support_layer))
        self.assertTrue(
            pqat_support_layer
            in self.default_8bit_quantize_registry._layer_quantize_map,
            msg="Default 8bit QAT doesn't support {}".format(
                pqat_support_layer))


if __name__ == '__main__':
  tf.test.main()
