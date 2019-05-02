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
"""A Keras wrapper to add pruning related variables to a layer."""

# pylint: disable=missing-docstring,g-multiple-import,unused-import,protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import g3
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.wrappers import Wrapper
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer
from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_impl
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched


class PruneLowMagnitude(Wrapper):
  """This wrapper augments a keras layer so the weight tensor may be pruned.

  This wrapper implements magnitude-based pruning of the weight tensors.
  Magnitude-based pruning achieves a target sparsity (s% of zeros) for a given
  weight tensor by monitoring the distribution of the absolute values of the
  weight tensor and determining the weight value (referred to as threshold)
  below which s% of elements lie. For every weight tensor being pruned, the
  wrapper maintains an identically shaped tensor (referred to as mask) which
  stores 0 if the weight value lies below the threshold.
  The mask and thresholds are computed during the training based on the
  evolution of the weight values.

  Block sparse patterns:
  For certain SIMD hardware architectures, it may be beneficial to induce
  spatially correlated sparsity. To train models in which the weight tensors
  have block sparse structure, the pruning wrapper can be configured with
  the block_height and block_width configuration parameters set to the desired
  block configuration (2x2, 4x4, 4x1, 1x8, etc). This is applicable to
  rank-2 weight tensor only and the tensor partitioned into non-overlapping
  blocks of size [block_height, block_dim]. Either the average or max absolute
  value in this block is taken as a proxy for the entire block
  (set by block_pooling_function configuration parameter)
  while computing the distribution of the weight values and
  the threshold for pruning.

  Custom keras layers:
  The pruning wrapper can also be applied to a user-defined keras layer.
  Such a layer may contain one or more weight tensors that may be pruned.
  To apply pruning wrapper to such layers, set prunable_weight_names to mark
  the weight tensors for pruning.

  Sparsity function:
  The target sparsity for the weight tensors are set through the
  pruning_schedule parameter of the pruning wrapper. The user must create a
  python callable that returns a scalar tensorflow tensor and pass this
  callable to the sparsity_function parameter. This scalar tensor contains the
  target sparsity value for the weight tensors in the layer.
  The wrapper provides the following pre-built sparsity functions:

  ConstantSparsity
  GradualSparsity

  Eg.
  params = PruningParams(frequency=10,pruning_schedule=ConstantSparsity(0.9))
  pruned_model = keras.model.Sequential()
  pruned_model.add(
      Prune(keras.layers.Dense(256), input_shape=(256,)))
  pruned_model.add(Prune(keras.layers.Dense(1024), params=params))

  """

  _PRUNE_CALLBACK_ERROR_MSG = (
      'Prune() wrapper requires the UpdatePruningStep callback to be provided '
      'during training. Please add it as a callback to your model.fit call.')

  def __init__(self,
               layer,
               pruning_schedule=pruning_sched.ConstantSparsity(0.5, 0),
               block_size=(1, 1),
               block_pooling_type='AVG',
               **kwargs):
    """Create a pruning wrapper for a keras layer.

    #TODO(pulkitb): Consider if begin_step should be 0 by default.

    Args:
      layer: The keras layer to be pruned.
      pruning_schedule: A `PruningSchedule` object that controls pruning rate
        throughout training.
      block_size: (optional) The dimensions (height, weight) for the block
        sparse pattern in rank-2 weight tensors.
      block_pooling_type: (optional) The function to use to pool weights in the
        block. Must be 'AVG' or 'MAX'.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    self.pruning_schedule = pruning_schedule
    self.block_size = block_size
    self.block_pooling_type = block_pooling_type

    # An instance of the Pruning class. This class contains the logic to prune
    # the weights of this layer.
    self.pruning_obj = None

    # A list of all (weight,mask,threshold) tuples for this layer
    self.pruning_vars = []

    if block_pooling_type not in ['AVG', 'MAX']:
      raise ValueError(
          'Unsupported pooling type \'{}\'. Should be \'AVG\' or \'MAX\'.'
          .format(block_pooling_type))

    if not isinstance(layer, Layer):
      raise ValueError(
          'Please initialize `Prune` layer with a '
          '`Layer` instance. You passed: {input}'.format(input=layer))

    # TODO(pulkitb): This should be pushed up to the wrappers.py
    # Name the layer using the wrapper and underlying layer name.
    # Prune(Dense) becomes prune_dense_1
    kwargs.update({'name': '{}_{}'.format(
        generic_utils.to_snake_case(self.__class__.__name__), layer.name)})

    if isinstance(layer, prunable_layer.PrunableLayer):
      # Custom layer in client code which supports pruning.
      super(PruneLowMagnitude, self).__init__(layer, **kwargs)
    elif prune_registry.PruneRegistry.supports(layer):
      # Built-in keras layers which support pruning.
      super(PruneLowMagnitude, self).__init__(
          prune_registry.PruneRegistry.make_prunable(layer), **kwargs)
    else:
      raise ValueError(
          'Please initialize `Prune` with a supported layer. Layers should '
          'either be a `PrunableLayer` instance, or should be supported by the '
          'PruneRegistry. You passed: {input}'.format(input=layer.__class__))

    self._track_trackable(layer, name='layer')

    # TODO(yunluli): Work-around to handle the first layer of Sequential model
    # properly. Can remove this when it is implemented in the Wrapper base
    # class.
    # The _batch_input_shape attribute in the first layer makes a Sequential
    # model to be built. This change makes sure that when we apply the wrapper
    # to the whole model, this attribute is pulled into the wrapper to preserve
    # the 'built' state of the model.
    if not hasattr(self, '_batch_input_shape') and hasattr(
        layer, '_batch_input_shape'):
      self._batch_input_shape = self.layer._batch_input_shape

  def build(self, input_shape):
    super(PruneLowMagnitude, self).build(input_shape)

    weight_vars, mask_vars, threshold_vars = [], [], []

    self.prunable_weights = self.layer.get_prunable_weights()

    # For each of the prunable weights, add mask and threshold variables
    for weight in self.prunable_weights:
      mask = self.add_variable(
          'mask',
          shape=weight.shape,
          initializer=initializers.get('ones'),
          dtype=weight.dtype,
          trainable=False,
          aggregation=tf_variables.VariableAggregation.MEAN)
      threshold = self.add_variable(
          'threshold',
          shape=[],
          initializer=initializers.get('zeros'),
          dtype=weight.dtype,
          trainable=False,
          aggregation=tf_variables.VariableAggregation.MEAN)

      weight_vars.append(weight)
      mask_vars.append(mask)
      threshold_vars.append(threshold)
    self.pruning_vars = list(zip(weight_vars, mask_vars, threshold_vars))

    # Add a scalar tracking the number of updates to the wrapped layer.
    self.pruning_step = self.add_variable(
        'pruning_step',
        shape=[],
        initializer=initializers.Constant(-1),
        dtype=dtypes.int32,
        trainable=False)

    def training_step_fn():
      return self.pruning_step

    # Create a pruning object
    self.pruning_obj = pruning_impl.Pruning(
        training_step_fn=training_step_fn,
        pruning_vars=self.pruning_vars,
        pruning_schedule=self.pruning_schedule,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    def add_update():
      with ops.control_dependencies([
          check_ops.assert_greater_equal(
              self.pruning_step, 0, message=self._PRUNE_CALLBACK_ERROR_MSG)]):
        with ops.control_dependencies(
            [self.pruning_obj.conditional_mask_update()]):
          return control_flow_ops.no_op('update')

    def no_op():
      return control_flow_ops.no_op('no_update')

    update_op = tf_utils.smart_cond(training, add_update, no_op)
    self.add_update(update_op)
    # Always execute the op that performs weights = weights * mask
    self.add_update(self.pruning_obj.weight_mask_op())

    return self.layer.call(inputs)

  def compute_output_shape(self, input_shape):
    return self.layer.compute_output_shape(input_shape)

  def get_config(self):
    base_config = super(PruneLowMagnitude, self).get_config()
    config = {
        'pruning_schedule': self.pruning_schedule.get_config(),
        'block_size': self.block_size,
        'block_pooling_type': self.block_pooling_type
    }
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()

    pruning_schedule = config.pop('pruning_schedule')
    from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object  # pylint: disable=g-import-not-at-top
    # TODO(pulkitb): This should ideally be fetched from pruning_schedule,
    # which should maintain a list of all the pruning_schedules.
    custom_objects = {
        'ConstantSparsity': pruning_sched.ConstantSparsity,
        'PolynomialDecay': pruning_sched.PolynomialDecay
    }
    config['pruning_schedule'] = deserialize_keras_object(
        pruning_schedule,
        module_objects=globals(),
        custom_objects=custom_objects)

    from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    layer = deserialize_layer(config.pop('layer'))
    config['layer'] = layer

    return cls(**config)

  @property
  def trainable(self):
    return self.layer.trainable

  @trainable.setter
  def trainable(self, value):
    self.layer.trainable = value

  @property
  def trainable_weights(self):
    return self.layer.trainable_weights

  @property
  def non_trainable_weights(self):
    return self.layer.non_trainable_weights + self._non_trainable_weights

  @property
  def updates(self):
    return self.layer.updates + self._updates

  @property
  def losses(self):
    return self.layer.losses + self._losses

  def get_weights(self):
    return self.layer.get_weights()

  def set_weights(self, weights):
    self.layer.set_weights(weights)
