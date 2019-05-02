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
"""End to End tests for the Pruning API."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import layers
from tensorflow.python.platform import test
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import test_utils


@tf_test_util.run_all_in_graph_and_eager_modes
class PruneIntegrationTest(test.TestCase, parameterized.TestCase):

  # Fetch all the prunable layers from the registry.
  _PRUNABLE_LAYERS = [
      layer for layer, weights in
      prune_registry.PruneRegistry._LAYERS_WEIGHTS_MAP.items() if weights
  ]

  # Fetch all the non-prunable layers from the registry.
  _NON_PRUNABLE_LAYERS = [
      layer for layer, weights in
      prune_registry.PruneRegistry._LAYERS_WEIGHTS_MAP.items()
      if not weights
  ]

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

  @staticmethod
  def _get_params_for_layer(layer_type):
    """Returns the arguments required to construct a layer.

    For a given `layer_type`, return ( [params], (input_shape) )

    Args:
      layer_type: Type of layer, Dense, Conv2D etc.

    Returns:
      Arguments to construct layer as a 2-tuple. First value is the list of
      params required to construct the layer. Second value is the input_shape
      for the layer.
    """
    return {
        layers.convolutional.Conv1D: ([4, 2], (3, 6)),
        layers.convolutional.Conv2D: ([4, (2, 2)], (4, 6, 1)),
        layers.convolutional.Conv2DTranspose: ([2, (3, 3)], (7, 6, 3)),
        layers.convolutional.Conv3D: ([2, (3, 3, 3)], (5, 7, 6, 3)),
        layers.convolutional.Conv3DTranspose: ([2, (3, 3, 3)], (5, 7, 6, 3)),
        layers.convolutional.SeparableConv1D: ([4, 3], (3, 6)),
        layers.convolutional.SeparableConv2D: ([4, (2, 2)], (4, 6, 1)),

        layers.core.Dense: ([4], (6,)),

        layers.local.LocallyConnected1D: ([4, 2], (3, 6)),
        layers.local.LocallyConnected2D: ([4, (2, 2)], (4, 6, 1)),

        # Embedding has a separate test since training it is not
        # feasible as a single layer.
        layers.embeddings.Embedding: (None, None),
    }[layer_type]

  def setUp(self):
    super(PruneIntegrationTest, self).setUp()
    self.params = {
        'pruning_schedule': pruning_schedule.ConstantSparsity(0.5, 0, -1, 1),
        # TODO(pulkitb): Add tests for block sparsity.
        'block_size': (1, 1),
        'block_pooling_type': 'AVG'
    }

  # TODO(pulkitb): Also assert correct weights are pruned.

  def _check_strip_pruning_matches_original(self, model, sparsity):
    stripped_model = prune.strip_pruning(model)
    test_utils.assert_model_sparsity(self, sparsity, stripped_model)

    input_data = np.random.randn(
        *self._batch(model.input.get_shape().as_list(), 1))

    model_result = model.predict(input_data)
    stripped_model_result = stripped_model.predict(input_data)
    np.testing.assert_almost_equal(model_result, stripped_model_result)

  @parameterized.parameters(_PRUNABLE_LAYERS)
  def testPrunesSingleLayer(self, layer_type):
    model = keras.Sequential()
    args, input_shape = self._get_params_for_layer(layer_type)
    if args is None:
      return  # Test for layer not supported yet.
    model.add(prune.prune_low_magnitude(
        layer_type(*args), input_shape=input_shape, **self.params))

    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    test_utils.assert_model_sparsity(self, 0.0, model)
    model.fit(
        np.random.randn(*self._batch(model.input.get_shape().as_list(), 32)),
        np.random.randn(*self._batch(model.output.get_shape().as_list(), 32)),
        callbacks=[pruning_callbacks.UpdatePruningStep()])

    test_utils.assert_model_sparsity(self, 0.5, model)

    self._check_strip_pruning_matches_original(model, 0.5)

  @parameterized.parameters(
      prune_registry.PruneRegistry._RNN_LAYERS - {keras.layers.RNN})
  def testRNNLayersSingleCell(self, layer_type):
    model = keras.Sequential()
    model.add(
        prune.prune_low_magnitude(
            layer_type(10), input_shape=(3, 4), **self.params))

    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    test_utils.assert_model_sparsity(self, 0.0, model)
    model.fit(
        np.random.randn(*self._batch(model.input.get_shape().as_list(), 32)),
        np.random.randn(*self._batch(model.output.get_shape().as_list(), 32)),
        callbacks=[pruning_callbacks.UpdatePruningStep()])

    test_utils.assert_model_sparsity(self, 0.5, model)

    self._check_strip_pruning_matches_original(model, 0.5)

  def testRNNLayersWithRNNCellParams(self):
    model = keras.Sequential()
    model.add(
        prune.prune_low_magnitude(
            keras.layers.RNN([
                layers.LSTMCell(10),
                layers.GRUCell(10),
                layers.PeepholeLSTMCell(10),
                layers.SimpleRNNCell(10)
            ]),
            input_shape=(3, 4),
            **self.params))

    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    test_utils.assert_model_sparsity(self, 0.0, model)
    model.fit(
        np.random.randn(*self._batch(model.input.get_shape().as_list(), 32)),
        np.random.randn(*self._batch(model.output.get_shape().as_list(), 32)),
        callbacks=[pruning_callbacks.UpdatePruningStep()])

    test_utils.assert_model_sparsity(self, 0.5, model)

    self._check_strip_pruning_matches_original(model, 0.5)

  def testPrunesEmbedding(self):
    model = keras.Sequential()
    model.add(
        prune.prune_low_magnitude(
            keras.layers.Embedding(input_dim=10, output_dim=3),
            input_shape=(5,),
            **self.params))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    test_utils.assert_model_sparsity(self, 0.0, model)
    model.fit(
        np.random.randint(10, size=(32, 5)),
        np.random.randint(2, size=(32, 1)),
        callbacks=[pruning_callbacks.UpdatePruningStep()])

    test_utils.assert_model_sparsity(self, 0.5, model)

    self._check_strip_pruning_matches_original(model, 0.5)

  @parameterized.parameters(test_utils.model_type_keys())
  def testPrunesModel(self, model_type):
    model = test_utils.build_mnist_model(model_type, self.params)
    if model_type == 'layer_list':
      model = keras.Sequential(prune.prune_low_magnitude(model, **self.params))
    elif model_type in ['sequential', 'functional']:
      model = prune.prune_low_magnitude(model, **self.params)

    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    test_utils.assert_model_sparsity(self, 0.0, model)
    model.fit(
        np.random.rand(32, 28, 28, 1),
        keras.utils.to_categorical(np.random.randint(10, size=(32, 1)), 10),
        callbacks=[pruning_callbacks.UpdatePruningStep()])

    test_utils.assert_model_sparsity(self, 0.5, model)

    self._check_strip_pruning_matches_original(model, 0.5)

  @parameterized.parameters(test_utils.save_restore_fns())
  def testPruneStopAndRestartOnModel(self, save_restore_fn):
    params = {
        'pruning_schedule': pruning_schedule.PolynomialDecay(
            0.2, 0.6, 0, 4, 3, 1),
        'block_size': (1, 1),
        'block_pooling_type': 'AVG'
    }
    model = prune.prune_low_magnitude(
        test_utils.build_simple_dense_model(), **params)
    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # Model hasn't been trained yet. Sparsity 0.0
    test_utils.assert_model_sparsity(self, 0.0, model)

    model.fit(
        np.random.rand(20, 10),
        keras.utils.to_categorical(np.random.randint(5, size=(20, 1)), 5),
        batch_size=20,
        callbacks=[pruning_callbacks.UpdatePruningStep()])
    # Training has run only 1 step. Sparsity 0.2 (initial_sparsity)
    test_utils.assert_model_sparsity(self, 0.2, model)

    model = save_restore_fn(model)
    model.fit(
        np.random.rand(20, 10),
        keras.utils.to_categorical(np.random.randint(5, size=(20, 1)), 5),
        batch_size=20,
        epochs=3,
        callbacks=[pruning_callbacks.UpdatePruningStep()])
    # Training has run all 4 steps. Sparsity 0.6 (final_sparsity)
    test_utils.assert_model_sparsity(self, 0.6, model)

    self._check_strip_pruning_matches_original(model, 0.6)

  @parameterized.parameters(test_utils.save_restore_fns())
  def testPruneWithPolynomialDecayPreservesSparsity(self, save_restore_fn):
    params = {
        'pruning_schedule': pruning_schedule.PolynomialDecay(
            0.2, 0.6, 0, 1, 3, 1),
        'block_size': (1, 1),
        'block_pooling_type': 'AVG'
    }
    model = prune.prune_low_magnitude(
        test_utils.build_simple_dense_model(), **params)
    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # Model hasn't been trained yet. Sparsity 0.0
    test_utils.assert_model_sparsity(self, 0.0, model)

    model.fit(
        np.random.rand(20, 10),
        keras.utils.to_categorical(np.random.randint(5, size=(20, 1)), 5),
        batch_size=20,
        callbacks=[pruning_callbacks.UpdatePruningStep()])
    # Training has run only 1 step. Sparsity 0.2 (initial_sparsity)
    test_utils.assert_model_sparsity(self, 0.2, model)

    model.fit(
        np.random.rand(20, 10),
        keras.utils.to_categorical(np.random.randint(5, size=(20, 1)), 5),
        batch_size=20,
        callbacks=[pruning_callbacks.UpdatePruningStep()])
    # Training has run 2 steps. Sparsity 0.6 (final_sparsity)
    test_utils.assert_model_sparsity(self, 0.6, model)

    model = save_restore_fn(model)
    model.fit(
        np.random.rand(20, 10),
        keras.utils.to_categorical(np.random.randint(5, size=(20, 1)), 5),
        batch_size=20,
        epochs=2,
        callbacks=[pruning_callbacks.UpdatePruningStep()])
    # Training has run all 4 steps. Sparsity 0.6 (final_sparsity)
    test_utils.assert_model_sparsity(self, 0.6, model)

    self._check_strip_pruning_matches_original(model, 0.6)

  def testPrunesPreviouslyUnprunedModel(self):
    model = test_utils.build_simple_dense_model()
    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # Simple unpruned model. No sparsity.
    model.fit(
        np.random.rand(20, 10),
        keras.utils.to_categorical(np.random.randint(5, size=(20, 1)), 5),
        epochs=2,
        batch_size=20)
    test_utils.assert_model_sparsity(self, 0.0, model)

    # Apply pruning to model.
    model = prune.prune_low_magnitude(model, **self.params)
    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # Since newly compiled, iterations starts from 0.
    model.fit(
        np.random.rand(20, 10),
        keras.utils.to_categorical(np.random.randint(5, size=(20, 1)), 5),
        batch_size=20,
        callbacks=[pruning_callbacks.UpdatePruningStep()])
    test_utils.assert_model_sparsity(self, 0.5, model)

    self._check_strip_pruning_matches_original(model, 0.5)


if __name__ == '__main__':
  test.main()
