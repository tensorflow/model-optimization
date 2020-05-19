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

import tempfile
import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

# TODO(b/139939526): move to public API.
from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.keras import test_utils as keras_test_utils
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.python.core.sparsity.keras import test_utils


keras = tf.keras
layers = keras.layers

list_to_named_parameters = test_utils.list_to_named_parameters
ModelCompare = keras_test_utils.ModelCompare


@keras_parameterized.run_all_keras_modes
class PruneIntegrationTest(tf.test.TestCase, parameterized.TestCase,
                           ModelCompare):

  # Fetch all the prunable layers from the registry.
  _PRUNABLE_LAYERS = [
      layer for layer, weights in
      prune_registry.PruneRegistry._LAYERS_WEIGHTS_MAP.items()
        if (weights and layer != tf.keras.layers.Conv3DTranspose
                    and layer != tf.keras.layers.Conv2DTranspose)
  ]

  # Fetch all the non-prunable layers from the registry.
  _NON_PRUNABLE_LAYERS = [
      layer for layer, weights in
      prune_registry.PruneRegistry._LAYERS_WEIGHTS_MAP.items()
      if not weights
  ]

  @staticmethod
  def _save_as_saved_model(model):
    saved_model_dir = tempfile.mkdtemp()
    model.save(saved_model_dir)
    return saved_model_dir

  @staticmethod
  def get_gzipped_model_size(model):
    # It returns the size of the gzipped model in bytes.
    import os
    import zipfile

    _, keras_file = tempfile.mkstemp('.h5')
    model.save_weights(keras_file)

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
      f.write(keras_file)
    return os.path.getsize(zipped_file)


  @staticmethod
  def _get_directory_size_in_bytes(directory):
    import os

    total = 0
    try:
        for entry in os.scandir(directory):
            if entry.is_file():
                # if it's a file, use stat() function
                total += entry.stat().st_size
            elif entry.is_dir():
                # if it's a directory, recursively call this function
                total += PruneIntegrationTest._get_directory_size_in_bytes(entry.path)
    except NotADirectoryError:
        # if `directory` isn't a directory, get the file size then
        return os.path.getsize(directory)
    except PermissionError:
        # if for whatever reason we can't open the folder, return 0
        return 0
    return total

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
        layers.Conv1D: ([4, 2], (3, 6)),
        layers.Conv2D: ([4, (2, 2)], (4, 6, 1)),
        # TODO(tf-mot): fix for Conv2DTranspose on some form of eager,
        # with or without functions. The weights become nan (though the
        # mask seems fine still).
        #layers.Conv2DTranspose: ([2, (3, 3)], (7, 6, 3)),
        layers.Conv3D: ([2, (3, 3, 3)], (5, 7, 6, 3)),
        # TODO(tf-mot): fix for Conv3DTranspose on some form of eager,
        # with or without functions. The weights become nan (though the
        # mask seems fine still).
        #layers.Conv3DTranspose: ([2, (3, 3, 3)], (5, 7, 6, 3)),
        layers.SeparableConv1D: ([4, 3], (3, 6)),
        layers.SeparableConv2D: ([4, (2, 2)], (4, 6, 1)),
        layers.Dense: ([4], (6,)),
        layers.LocallyConnected1D: ([4, 2], (3, 6)),
        layers.LocallyConnected2D: ([4, (2, 2)], (4, 6, 1)),

        # Embedding has a separate test since training it is not
        # feasible as a single layer.
        layers.Embedding: (None, None),
    }[layer_type]

  def setUp(self):
    super(PruneIntegrationTest, self).setUp()
    self.params = {
        'pruning_schedule': pruning_schedule.ConstantSparsity(0.5, 0, -1, 1),
        # TODO(pulkitb): Add tests for block sparsity.
        'block_size': (1, 1),
        'block_pooling_type': 'AVG'
    }

  @staticmethod
  def _is_pruned(model):
    for layer in model.layers:
      if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
        return True

  @staticmethod
  def _train_model(model, epochs=1, x_train=None, y_train=None, callbacks=None):
    if x_train is None:
      x_train = np.random.rand(20, 10)
    if y_train is None:
      y_train = keras.utils.to_categorical(
          np.random.randint(5, size=(20, 1)), 5)

    if model.optimizer is None:
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])

    if callbacks is None:
      callbacks = []
      if PruneIntegrationTest._is_pruned(model):
        callbacks = [pruning_callbacks.UpdatePruningStep()]

    model.fit(
        x_train, y_train, epochs=epochs, batch_size=20, callbacks=callbacks)

  @staticmethod
  def _get_subclassed_model():

    class TestSubclassedModel(keras.Model):
      """A model subclass."""

      def __init__(self):
        """A test subclass model with one dense layer."""
        super(TestSubclassedModel, self).__init__(name='test_model')
        self.layer1 = keras.layers.Dense(8, activation='relu')
        self.layer2 = keras.layers.Dense(5, activation='softmax')

      def call(self, inputs):
        x = self.layer1(inputs)
        return self.layer2(x)

    return TestSubclassedModel()


  def testPrunePretrainedSubclassedModelAttributes_WrapperAdded_SizeIncreases(
      self):
    # Size increases since wrapper adds new weights and we don't call
    # `strip_pruning`.

    model = self._get_subclassed_model()
    self._train_model(model, epochs=1)

    input_data = np.random.rand(10, 10)

    pruned_model = self._get_subclassed_model()

    # Build the model and copy weights over.
    pruned_model.build(input_data.shape)
    pruned_model.set_weights(model.get_weights())

    # Apply pruning.
    pruned_model.layer1 = prune.prune_low_magnitude(pruned_model.layer1,
                                                   **self.params)
    pruned_model.layer2 = prune.prune_low_magnitude(pruned_model.layer2,
                                                   **self.params)

    # Rebuild given added wrappers from `prune_low_magnitude`.
    pruned_model.build(input_data.shape)

    print("h5 weights:", self.get_gzipped_model_size(model))
    print("pruned h5 weights:", self.get_gzipped_model_size(pruned_model))

  def testPrunePretrainedSubclassedModelAttributes_WrapperAdded_CallChanged(
      self):

    model = self._get_subclassed_model()
    self._train_model(model, epochs=1)

    input_data = np.random.rand(10, 10)

    pruned_model = self._get_subclassed_model()

    # Build the model and copy weights over.
    pruned_model.build(input_data.shape)
    pruned_model.set_weights(model.get_weights())

    # Apply pruning.
    pruned_model.layer1 = prune.prune_low_magnitude(pruned_model.layer1,
                                                   **self.params)
    pruned_model.layer2 = prune.prune_low_magnitude(pruned_model.layer2,
                                                   **self.params)

    print("starting training: should throw debugging message from wrapper given that the UpdatePruningStep callback isn't being called.")
    self._train_model(pruned_model, epochs=1, callbacks=[])

  def testPrunePretrainedSubclassedModelAttributes_WrapperAddedAfterPredict_TrainingCallChanged(
      self):

    model = self._get_subclassed_model()
    self._train_model(model, epochs=1)

    input_data = np.random.rand(10, 10)

    pruned_model = self._get_subclassed_model()

    # Build the model and copy weights over.
    pruned_model.build(input_data.shape)
    pruned_model.set_weights(model.get_weights())

    # Call `call` to see if it makes it so that setting the attributes
    # no longer does anything.
    pruned_out = pruned_model.predict(input_data)

    # Apply pruning.
    print("applying wrappers after predict")
    pruned_model.layer1 = prune.prune_low_magnitude(pruned_model.layer1,
                                                   **self.params)
    pruned_model.layer2 = prune.prune_low_magnitude(pruned_model.layer2,
                                                   **self.params)

    print("starting training: if call changed, should throw debugging error from wrapper given that the UpdatePruningStep callback isn't being called.")
    self._train_model(pruned_model, epochs=1, callbacks=[])
    # Error is indeed thrown and print statements (not tf.Print) in wrapper's
    # `call` are still executed.

  def testPrunePretrainedSubclassedModelAttributes_WrapperAddedAfterPredict_PredictCallMaybeChanged(
      self):

    model = self._get_subclassed_model()
    self._train_model(model, epochs=1)

    input_data = np.random.rand(10, 10)

    pruned_model = self._get_subclassed_model()

    # Build the model and copy weights over.
    pruned_model.build(input_data.shape)
    pruned_model.set_weights(model.get_weights())

    # Call `call` to see if it makes it so that pruning isn't applied.
    pruned_out = pruned_model.predict(input_data)

    # Apply pruning.
    print("applying wrappers after predict")
    pruned_model.layer1 = prune.prune_low_magnitude(pruned_model.layer1,
                                                   **self.params)
    pruned_model.layer2 = prune.prune_low_magnitude(pruned_model.layer2,
                                                   **self.params)

    pruned_out = pruned_model.predict(input_data)
    # print statements in `call` no longer called.



# pruned_model.layer1 = prune.strip_pruning(pruned_model.layer1)
# pruned_model.layer2 = prune.strip_pruning(pruned_model.layer2)

# print("prune predict")
# out = model.predict(input_data)
# pruned_out = pruned_model.predict(input_data)
# self.assertTrue((out == pruned_out).all())

# print("tf weights:", self.get_gzipped_model_size(model))
# print("pruned tf weights:", self.get_gzipped_model_size(pruned_model))

# original_saved_model_dir = self._save_as_saved_model(model)
# saved_model_dir = self._save_as_saved_model(pruned_model)

# original_size = self._get_directory_size_in_bytes(original_saved_model_dir)
# compressed_size = self._get_directory_size_in_bytes(saved_model_dir)

# print("original size:", original_size)
# print("compressed size:", compressed_size)

if __name__ == '__main__':
  tf.test.main()
