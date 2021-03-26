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

from absl.testing import parameterized

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

from tensorflow.python.keras import keras_parameterized

from tensorflow_model_optimization.python.core.clustering.keras import cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.quantization.keras import quant_ops
from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras.collaborative_optimizations.cluster_preserve import (
    default_8bit_cluster_preserve_quantize_scheme,)

layers = tf.keras.layers

@keras_parameterized.run_all_keras_modes
class ClusterPreserveIntegrationTest(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        super(ClusterPreserveIntegrationTest, self).setUp()
        self.cluster_params = {
            "number_of_clusters": 4,
            "cluster_centroids_init": cluster_config.CentroidInitialization.LINEAR
        }

    def compile_and_fit(self, model):
        """ Here we compile and fit the model
        """
        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer="adam",
            metrics=["accuracy"],
        )
        model.fit(
            np.random.rand(20, 10),
            tf.keras.utils.to_categorical(np.random.randint(5, size=(20, 1)), 5),
            batch_size=20)

    def testEndToEndClusterPreserve(self):
        """Verifies that we can run CQAT end to end,
        when the whole model is quantized."""
        original_model = tf.keras.Sequential([
            layers.Dense(5, activation='relu', input_shape=(10,))
        ])

        clustered_model = cluster.cluster_weights(original_model,
            **self.cluster_params)

        self.compile_and_fit(clustered_model)

        clustered_model = cluster.strip_clustering(clustered_model)

        quant_aware_annotate_model = (
            quantize.quantize_annotate_model(clustered_model))

        quant_aware_model = quantize.quantize_apply(
            quant_aware_annotate_model,
            scheme=default_8bit_cluster_preserve_quantize_scheme
            .Default8BitClusterPreserveQuantizeScheme())

        self.compile_and_fit(quant_aware_model)

    def testEndToEndClusterPreservePerLayer(self):
        """Verifies that we can run CQAT end to end,
        when the model is quantized per layers"""
        original_model = tf.keras.Sequential([
            layers.Dense(5, activation='relu', input_shape=(10,)),
            layers.Dense(5, activation='relu', input_shape=(10,))
        ])

        clustered_model = cluster.cluster_weights(original_model,
            **self.cluster_params)

        self.compile_and_fit(clustered_model)

        clustered_model = cluster.strip_clustering(clustered_model)

        def apply_quantization_to_dense(layer):
            if isinstance(layer, tf.keras.layers.Dense):
                return quantize.quantize_annotate_layer(layer)
            return layer

        quant_aware_annotate_model = tf.keras.models.clone_model(clustered_model,
            clone_function=apply_quantization_to_dense,
        )

        quant_aware_model = quantize.quantize_apply(
            quant_aware_annotate_model,
            scheme=default_8bit_cluster_preserve_quantize_scheme
            .Default8BitClusterPreserveQuantizeScheme())

        self.compile_and_fit(quant_aware_model)

    def testEndToEndClusterPreserveOneLayer(self):
        """Verifies that we can run CQAT end to end,
        when the model is quantized per layers"""
        original_model = tf.keras.Sequential([
            layers.Dense(5, activation='relu', input_shape=(10,)),
            layers.Dense(5, activation='relu', input_shape=(10,), name="qat")
        ])

        clustered_model = cluster.cluster_weights(original_model,
            **self.cluster_params)

        self.compile_and_fit(clustered_model)

        clustered_model = cluster.strip_clustering(clustered_model)

        def apply_quantization_to_dense(layer):
            if isinstance(layer, tf.keras.layers.Dense):
                if (layer.name == "qat"):
                    return quantize.quantize_annotate_layer(layer)
            return layer

        quant_aware_annotate_model = tf.keras.models.clone_model(clustered_model,
            clone_function=apply_quantization_to_dense,
        )

        quant_aware_model = quantize.quantize_apply(
            quant_aware_annotate_model,
            scheme=default_8bit_cluster_preserve_quantize_scheme
            .Default8BitClusterPreserveQuantizeScheme())

        self.compile_and_fit(quant_aware_model)

if __name__ == '__main__':
  tf.test.main()
