# Lint as: python3
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
"""Tests for pruning_algorithm.py."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_model_optimization.python.core.common import algorithm
from tensorflow_model_optimization.python.core.common.keras.experimental import pruning_algorithm
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule

K = tf.keras.backend

_TEST_DATA_SIZE = 100


class LowMagnitudePrunerTest(tf.test.TestCase):

  def test_initial_state(self):
    """Tests initial_state method."""
    pruner = _get_test_pruner()
    weights = [tf.constant([1., 2., 3., 4., 5.])]
    state = pruner.initial_state(weights)
    expected_state = algorithm.StateRepr(
        {'weights': [tf.constant([1., 2., 3., 4., 5.])]}, {
            'masks': [tf.ones(5)],
            'step': tf.constant(0)
        })
    self._assert_state_equal(expected_state, state)

  def test_get_weight_tensors(self):
    """Tests get_weight_tensors method."""
    pruner = _get_test_pruner()
    state = algorithm.StateRepr(
        {'weights': [tf.constant([1., 2., 3., 4., 5.])]}, {
            'masks': [tf.constant([1., 1., 0., 0., 1.])],
            'step': tf.constant(0)
        })
    weights = pruner.get_weight_tensors(state)[0]
    expected_weights = tf.constant([1., 2., 0., 0., 5.])
    self.assertAllEqual(expected_weights, weights)

  def test_update_state(self):
    """Tests update_state method."""
    pruner = _get_test_pruner()

    # Before step 100, pruning should not be applied.
    state = algorithm.StateRepr(
        {'weights': [tf.constant([1., 2., 3., 4., 5.])]}, {
            'masks': [tf.ones(5)],
            'step': tf.constant(0)
        })
    state = pruner.update_state(state)
    expected_state = algorithm.StateRepr(
        {'weights': [tf.constant([1., 2., 3., 4., 5.])]}, {
            'masks': [tf.ones(5)],
            'step': tf.constant(1)
        })
    self._assert_state_equal(expected_state, state)

    state = algorithm.StateRepr(
        {'weights': [tf.constant([1., 2., 3., 4., 5.])]}, {
            'masks': [tf.ones(5)],
            'step': tf.constant(100)
        })
    state = pruner.update_state(state)
    expected_state = algorithm.StateRepr(
        {'weights': [tf.constant([0., 0., 3., 4., 5.])]}, {
            'masks': [tf.constant([0., 0., 1., 1., 1.])],
            'step': tf.constant(101)
        })
    self._assert_state_equal(expected_state, state)

  def test_add_loss_none(self):
    """Tests extra_loss is not provided."""
    pruner = _get_test_pruner()
    state = pruner.initial_state([tf.ones(5)])
    self.assertIsNone(pruner.add_loss(state))

  def test_illegal_input_raises(self):
    with self.assertRaisesRegex(ValueError, 'PruningSchedule'):
      pruning_algorithm.LowMagnitudePruner(0.0)

  # TODO(konkey): This could be a shared utility.
  def _assert_state_equal(self, expected_state, state):
    """Utility for testing equality of algorithm.StateRepr objects."""
    tf.nest.assert_same_structure(expected_state, state)
    for elem in zip(tf.nest.flatten(expected_state), tf.nest.flatten(state)):
      self.assertAllEqual(elem[0], elem[1])


def _get_test_pruner():
  schedule = pruning_schedule.ConstantSparsity(
      target_sparsity=0.4, begin_step=100, end_step=200, frequency=1)
  return pruning_algorithm.LowMagnitudePruner(schedule)


class PruneTrainingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('20_percent', 1),
      ('40_percent', 2),
      ('60_percent', 3),
      ('80_percent', 4),
  )
  def test_prune_dense_constant_schedule(self, target_nonzeros):
    """Tests that pruning works as expected with constant schedule."""
    with tf.Graph().as_default():  # TODO(konkey): Fix eager execution.
      schedule = pruning_schedule.ConstantSparsity(
          target_sparsity=target_nonzeros / 5,
          begin_step=100,
          end_step=-1,
          frequency=1)

      model, dataset = _test_dense_problem()
      model = pruning_algorithm.prune_low_magnitude(model, schedule)
      model.compile(
          optimizer=tf.keras.optimizers.SGD(),
          loss=tf.keras.losses.MeanSquaredError())

      model.fit(dataset, epochs=1)
      # Pruning did not yet kick in.
      self._assert_sparsity(
          model,
          num_nonzeros_expected=0,
          num_steps_expected=_TEST_DATA_SIZE)
      # Pruning should kick in the next step.
      model.fit(dataset, epochs=1, steps_per_epoch=1)
      self._assert_sparsity(
          model,
          num_nonzeros_expected=target_nonzeros,
          num_steps_expected=_TEST_DATA_SIZE + 1)

  def test_prune_dense_polynomial_decay_schedule(self):
    """Tests that pruning works as expected with polynomial schedule."""
    with tf.Graph().as_default():  # TODO(konkey): Fix eager execution.
      schedule = pruning_schedule.PolynomialDecay(
          initial_sparsity=0.0,
          final_sparsity=0.8,
          begin_step=99,
          end_step=109,
          power=3,
          frequency=1)

      model, dataset = _test_dense_problem()
      model = pruning_algorithm.prune_low_magnitude(model, schedule)
      model.compile(
          optimizer=tf.keras.optimizers.SGD(),
          loss=tf.keras.losses.MeanSquaredError())

      model.fit(dataset, epochs=1)
      # Target sparsity =0.00 -> 0/5 zeroed-out.
      self._assert_sparsity(
          model, num_nonzeros_expected=0, num_steps_expected=_TEST_DATA_SIZE)

      model.fit(dataset, epochs=1, steps_per_epoch=1)
      # Target sparsity ~0.21 -> 1/5 zeroed-out.
      self._assert_sparsity(
          model,
          num_nonzeros_expected=1,
          num_steps_expected=_TEST_DATA_SIZE + 1)

      model.fit(dataset, epochs=1, steps_per_epoch=1)
      # Target sparsity ~0.39 -> 2/5 zeroed-out.
      self._assert_sparsity(
          model,
          num_nonzeros_expected=2,
          num_steps_expected=_TEST_DATA_SIZE + 2)

      model.fit(dataset, epochs=1, steps_per_epoch=1)
      # Target sparsity ~0.53 -> 3/5 zeroed-out.
      self._assert_sparsity(
          model,
          num_nonzeros_expected=3,
          num_steps_expected=_TEST_DATA_SIZE + 3)

      model.fit(dataset, epochs=1, steps_per_epoch=3)
      # Target sparsity ~0.75 -> 4/5 zeroed-out.
      self._assert_sparsity(
          model,
          num_nonzeros_expected=4,
          num_steps_expected=_TEST_DATA_SIZE + 6)

  def test_prune_dense_correctly_zeros_out_gradients(self):
    """Tests that gradients are computed with respect to the masked model."""
    with tf.Graph().as_default():  # TODO(konkey): Fix eager execution.
      schedule = pruning_schedule.ConstantSparsity(
          target_sparsity=0.6,
          begin_step=100,
          end_step=100,
          frequency=1)

      model, dataset = _test_dense_problem()
      model = pruning_algorithm.prune_low_magnitude(model, schedule)
      model.compile(
          optimizer=tf.keras.optimizers.SGD(),
          loss=tf.keras.losses.MeanSquaredError())

      model.fit(dataset, epochs=1)
      # Pruning did not yet kick in.
      self._assert_sparsity(
          model,
          num_nonzeros_expected=0,
          num_steps_expected=_TEST_DATA_SIZE)

      model.fit(dataset, epochs=1)
      # Pruning was applied only from step #100 to #101, yet after training
      # progresses, the weights stay zeroed-out.
      self._assert_sparsity(
          model,
          num_nonzeros_expected=3,
          num_steps_expected=_TEST_DATA_SIZE * 2)

  def _assert_sparsity(self, model, num_nonzeros_expected, num_steps_expected):
    """Asserts state representation is as expected."""
    layer = model.layers[0]  # Only a single Dense layer present.
    state = tf.nest.map_structure(K.get_value, layer._state_vars)
    for mask in state.non_trainable['masks']:
      expected_zeros_part = mask[:num_nonzeros_expected]
      expected_ones_part = mask[num_nonzeros_expected:]
      self.assertAllEqual(
          np.zeros_like(expected_zeros_part), expected_zeros_part)
      self.assertAllEqual(np.ones_like(expected_ones_part), expected_ones_part)

    for weight in state.trainable['weights']:
      expected_zeros_part = weight[:num_nonzeros_expected]
      expected_nonzeros_part = weight[num_nonzeros_expected:]
      self.assertAllEqual(
          np.zeros_like(expected_zeros_part), expected_zeros_part)
      self.assertAllGreater(expected_nonzeros_part, 0.0)

    self.assertEqual(num_steps_expected, state.non_trainable['step'])


def _test_dense_problem():
  """Test problem using Dense Layer.

  The problem is designed such that the optimal solution is [1., 2., 3., 4., 5.]
  and thus after partial convergence, the sparsity pattern is fully predictable.

  Returns:
    A (model, dataset) tuple representing the problem.
  """

  def make_data(num_elements):
    w_star = np.array([[float(i + 1)] for i in range(5)])
    x = np.random.randn(num_elements, 5)
    y = np.matmul(x, w_star)
    return np.expand_dims(x, axis=1), y

  dataset = tf.data.Dataset.from_tensor_slices(make_data(_TEST_DATA_SIZE))

  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(1, input_shape=(5,)))

  return model, dataset


if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
