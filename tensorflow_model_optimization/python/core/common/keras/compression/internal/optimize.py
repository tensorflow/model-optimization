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
"""Internal APIs and core implementation of weight compression API."""
from typing import List, Mapping

import tensorflow as tf

from tensorflow_model_optimization.python.core.keras.compat import keras


# Workaround to prevent MLIR from constant folding the
# compressed weights into the original weights. For instance,
# if we decompose `self.kernel` into `u` and `v`, we need to
# make sure that decompression occurs during inference, instead
# of during MLIR optimization which could multiply `u` and `v`
# given that they are constants.
#
# TODO(tfmot): make this more stable. This currently relies
# on the TensorFlow Lite MLIR converter to not constant
# fold through `tf.cond`, even though it already does
# for `tf.while`.
def _prevent_constant_folding(tensor, dummy_inputs):
  tensor = tf.identity(tensor)
  outputs = tf.cond(
      tf.reduce_sum(dummy_inputs) > 0, lambda: tensor, lambda: tensor)
  return outputs


class _TrainingWrapper(keras.layers.Wrapper):
  """Represent modifications to training graph for weight compression."""

  def __init__(self, layer, algorithm, compressible_weights: List[str]):
    self.algorithm = algorithm
    self.compressible_weights = compressible_weights
    self.dummy_name_to_tensor = {}

    self.original_add_weight = layer.add_weight
    setattr(layer, 'add_weight', self._skip_compressible_weights)

    super(_TrainingWrapper, self).__init__(layer)

  # TODO(tfmot): We don't make tensor to map dict due to tensor is unhashable.
  def _get_name_by_tensor(self, tensor):
    for name, dummy_tensor in self.dummy_name_to_tensor.items():
      if tensor is dummy_tensor:
        return name
    return None

  def _skip_compressible_weights(self, *args, **kwargs):
    # Match for compressible weights based on `name` parameter.
    #
    # This depends on common practice where every layer's call
    # to `self.add_weight` follows this form:
    #
    #   self.`name` = self.add_weight(name=`name`)
    #
    # where the attribute name matches the variable name.
    #
    # TODO(tfmot): check if depending on this practice
    # is safe for both builtin and custom Keras layers.
    # Regardless, raise an exception if name is None, which
    # means that the practice has not been followed.
    name = None
    if args:
      name = args[0]
    if 'name' in kwargs:
      name = kwargs['name']

    if name not in self.compressible_weights:
      return self.original_add_weight(*args, **kwargs)

    # If weight is compressible, substitute in a dummy tensor
    # with the same shape as what would have been added.
    # Returning an empty tensor would cause ** to fail.
    shape = None
    if args and len(args) > 1:
      shape = args[1]
    if 'shape' in kwargs:
      shape = kwargs['shape']

    dummy_zeros = tf.zeros(shape)
    self.dummy_name_to_tensor[name] = dummy_zeros
    return dummy_zeros

  def build(self, input_shape):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    # Building nested layer via `super` must happen first
    # so that the nested layer's variables
    # are available to `init_training_weights`.
    super(_TrainingWrapper, self).build(input_shape)

    # Add weights needed by algorithm during training.
    self.training_weights = {}
    for name in self.compressible_weights:
      compressible_weight = self.dummy_name_to_tensor[name]

      # Note that as standard in `build` methods, the handling of pretrained
      # weights actually occurs outside the wrapper. This only initializes
      # weights with dummy values. Additionally, we don't have access to the
      # actual values of the nested layer's weights since they are no longer
      # variables, due to `_skip_compressible_weights` from `__init__`.
      assert isinstance(compressible_weight, tf.Tensor)
      self.algorithm.weight_reprs = []
      self.algorithm.init_training_weights(compressible_weight)

      weights = []
      for weight_repr in self.algorithm.weight_reprs:
        weight = self.add_weight(*weight_repr.args, **weight_repr.kwargs)
        weights.append(weight)

      self.training_weights[name] = weights

    self.attr_name_map = {}
    for attr_name in dir(self.layer):
      try:
        value = getattr(self.layer, attr_name)
        name = self._get_name_by_tensor(value)
      except AttributeError:
        # TODO(tfmot): Some attributes are not accessible like
        # 'input', 'input_mask', 'input_shape',
        # 'output', 'output_mask', 'output_shape'
        pass
      if name:
        self.attr_name_map[attr_name] = name

  def call(self, inputs):
    for attr_name, name in self.attr_name_map.items():
      # TODO(tfmot): move constant folding prevention to the inference graph
      # only, since constant folding won't happen during training.
      training_weight_tensors = []
      tensor_weight_pairs = []
      for v in self.training_weights[name]:
        tensor = _prevent_constant_folding(v.read_value(), inputs)
        training_weight_tensors.append(tensor)
        tensor_weight_pairs.append((tensor, v))

      self.algorithm.init_update_ops(tensor_weight_pairs)
      weight_tensor = self.algorithm.project_training_weights(
          *training_weight_tensors)
      # TODO(tfmot): Needs to check when this update is happen.
      self.add_update(self.algorithm.get_update_ops())
      setattr(self.layer, attr_name, weight_tensor)

    # This assumes that all changes to the forward pass happen "prior" to
    # the nested layer's portion of the forward pass. This suffices since
    # the scope of this API is to only optimize the weights.
    return self.layer.call(inputs)


# TODO(tfmot): deduplicate code with _TrainingWrapper.
class _InferenceWrapper(keras.layers.Wrapper):
  """Represent modifications to inference graph for weight compression."""

  def __init__(self, layer, algorithm,
               training_tensors: Mapping[str, List[tf.Tensor]]):
    self.algorithm = algorithm
    # training_tensors is a map from compressible attributes (e.g. 'kernel')
    # to tensors (not variables to prevent model size increasing) with the
    # same shape as the corresponding variables used during training.
    self.training_tensors = training_tensors

    self.original_add_weight = layer.add_weight
    setattr(layer, 'add_weight', self._skip_compressible_weights)

    super(_InferenceWrapper, self).__init__(layer)

  def _skip_compressible_weights(self, *args, **kwargs):
    # Match for compressible weights based on `name` parameter.
    #
    # This depends on common practice where every layer's call
    # to `self.add_weight` follows this form:
    #
    #   self.`name` = self.add_weight(name=`name`)
    #
    # where the attribute name matches the variable name.
    #
    # TODO(tfmot): check if depending on this practice
    # is safe for both builtin and custom Keras layers.
    # Regardless, raise an exception if name is None, which
    # means that the practice has not been followed.
    name = None
    if args:
      name = args[0]
    if 'name' in kwargs:
      name = kwargs['name']

    if name not in self.training_tensors:
      return self.original_add_weight(*args, **kwargs)

    # If weight is compressible, substitute in a dummy tensor
    # with the same shape as what would have been added.
    # Returning an empty tensor would cause ** to fail.
    shape = None
    if args and len(args) > 1:
      shape = args[1]
    if 'shape' in kwargs:
      shape = kwargs['shape']

    return tf.zeros(shape)

  def build(self, input_shape):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    super(_InferenceWrapper, self).build(input_shape)

    # Add weights needed by algorithm during inference.
    self.compressed_weights = {}
    for attr_name in self.training_tensors:
      training_tensors = self.training_tensors[attr_name]
      compressed_tensors = self.algorithm.compress_training_weights(
          *training_tensors)
      weights = []
      for t in compressed_tensors:
        weight = self.add_weight(
            name='TODO',
            dtype=t.dtype,
            shape=t.shape,
            initializer=keras.initializers.Constant(t),
        )
        weights.append(weight)

      self.compressed_weights[attr_name] = weights

  def call(self, inputs, training=None):
    for attr_name in self.training_tensors:
      # TODO(tfmot): understand how read_value() is converted to
      # inference in TensorFlow Lite.
      compressed_weight_tensors = []
      for v in self.compressed_weights[attr_name]:
        compressed_weight_tensors.append(
            _prevent_constant_folding(v.read_value(), inputs))
      weight_tensor = self.algorithm.decompress_weights(
          *compressed_weight_tensors)
      setattr(self.layer, attr_name, weight_tensor)

    # TODO(tfmot): handle training arg if needed given this is inference only.
    return self.layer.call(inputs)


def _find(value, items):
  for item in items:
    if value is item:
      return True
  return False


def _map_to_training_weights(
    algorithm,
    layer,
    compressible_weights: List[tf.Variable]) -> List[tf.Tensor]:
  """Construct the training weight values from the layer's pretrained weights.

    The weight values have the same structure as the output of
    `keras.layers.Layer.get_weights`.

  Args:
    algorithm: weight compression algorithm
    layer: layer
    compressible_weights: weight attributes of layer that should be compressed

  Returns:
    Initial weight values for training.
  """
  # Need to know for each layer that kernel is the first weight, etc.
  # TODO(tfmot): consider implication on custom Keras layers. The
  # user has to pass in the information that 'kernel' is the first
  # variable, 'bias' is the second variable, and so on.
  # TODO(tfmot): see if Keras can introduce changes to simplify this.
  original_weights = []
  training_weights = []
  if isinstance(layer, keras.layers.Conv2D) or isinstance(
      layer, keras.layers.Dense
  ):
    for weight in layer.weights:
      if _find(weight, compressible_weights):
        algorithm.weight_reprs = []
        algorithm.init_training_weights(weight)
        for weight_repr in algorithm.weight_reprs:
          # Assumes initializer is keras.initializers.Constant.
          # TODO(tfmot): add check for this assumption.
          # TODO(tfmot): the documentation for
          # keras.initializers.Constant(value)
          # suggests that the `value` cannot be any arbitrary shape and
          # only a single scalar value. It works in this implementation
          # to make `value` any tensor - check this.
          training_weights.append(weight_repr.kwargs['initializer'](
              shape=None, dtype=weight_repr.kwargs['dtype']))
      else:
        original_weights.append(weight)

  return training_weights + original_weights


# TODO(tfmot): same TODOs as _map_to_training_weights.
def _map_to_inference_weights(training_weights, algorithm, training_tensors):
  """Construct the inference weight values from the weights after training.

    The weight values have the same structure as the output of
    `keras.layers.Layer.get_weights`.

  Args:
    training_weights: layer's weights from training, retrieved via
      layer.get_weights()
    algorithm: weight compression algorithm
    training_tensors: map from compressible weight attribute (e.g. 'kernel') to
      relevant tensors.

  Returns:
    Initial weight values for training.

  Example:
    training_weights =    [kernel_var1, kernel_var2, bias]
    training_tensors = {'kernel': [kernel_var1, kernel_var2]}
    expected output:   [compress_training_weights(
      [kernel_var1, kernel_var2]), bias]
  """

  compressed_weights = []
  weights = ['kernel', 'bias']
  layer_weights_i = 0
  for weight in weights:
    if weight in training_tensors:
      compressed = algorithm.compress_training_weights(
          *training_tensors[weight])
      for c in compressed:
        compressed_weights.append(c.numpy())
      layer_weights_i += len(training_tensors[weight])
    else:
      compressed_weights.append(training_weights[layer_weights_i])
      layer_weights_i += 1
  return compressed_weights


def create_layer_for_training(layer, algorithm):
  """Internal API to create layer for training with weight compression."""

  # TODO(tfmot): move these checks to public API for
  # visibility.
  if not isinstance(algorithm, object):
    raise ValueError('`_create_layer_for_training` requires `algorithm` '
                     'to be an instantiated object, as opposed '
                     'to the class itself.')

  # Currently only supports a layer being built. The non-built
  # case may work fine as is, but it needs to be tested, together
  # with the followup API for exporting the model when the training
  # and inference graphs differ.
  if not layer.built:
    raise ValueError('`_create_layer_for_training` requires `layer` to '
                     'be built.')

  input_shape = layer.input_shape

  compressible_weights = algorithm.get_compressible_weights(layer)

  # Clone layer for two reasons:
  #
  #   1) Avoid unnecessary variable creation which undoes the benefits of
  #   compression. For instance, if we factorize `kernel` into `a` and `b`,
  #   since `a` and `b` collectively take less space than `kernel`, we
  #   no longer want to `kernel` to take up space as a variable.
  #
  #   The design depends on replacing the layer's `add_weight`
  #   method to prevent variable creation, before `add_weight` is called
  #   in the layer's `build`. Since the layer is built already, we undo
  #   this by cloning the layer.
  #
  #   2) The unoptimized layer and the optimized layer are now independent
  #   of each other and training one will not affect the other.
  #
  # TODO(tfmot): consider if it's okay to avoid this complexity during training
  # and only add it during inference, which is when model size really matters.
  # TODO(tfmot): handle custom Keras layer case.
  cloned_layer = layer.__class__.from_config(layer.get_config())

  # TODO(tfmot): We concider variable name form is layer_name/variable_name.
  layer_name = layer.name
  compressible_weights_name = []
  for compressible_weight in compressible_weights:
    name = compressible_weight.name
    if name.startswith(layer_name):
      name = name[len(layer_name)+1:]
    name = name.split(':')[0]
    compressible_weights_name.append(name)

  # TODO(tfmot): consider if this manner of handling build hinders
  # support for subclassed models in trying to set the attributes
  # that are layers while ensuring that the underlying trainable weights
  # have been created already.
  wrapped_layer = _TrainingWrapper(
      cloned_layer,
      algorithm,
      compressible_weights_name)

  if compressible_weights:
    # Set pretrained weight values.
    wrapped_layer.build(input_shape)
    # Clear `_build_input_shape` so that `build()` is not immediately called
    # during reloading. We want the wrapper layer to initiate `build()`.
    wrapped_layer.layer._build_input_shape = None  # pylint: disable=protected-access
    training_weights = _map_to_training_weights(
        algorithm,
        layer,
        compressible_weights)
    wrapped_layer.set_weights(
        [weight.numpy() for weight in training_weights])

  return wrapped_layer


def create_layer_for_inference(layer: _TrainingWrapper, algorithm):
  """Internal API to create layer for inference with weight compression."""
  # TODO(tfmot): move these checks to public API for
  # visibility.
  if not isinstance(algorithm, object):
    raise ValueError('`_create_layer_for_inference` requires `algorithm` '
                     'to be an instantiated object, as opposed '
                     'to the class itself.')

  if not layer.built:
    raise ValueError(
        '`_create_layer_for_inference` requires `layer` to be built.')

  # Process layer.
  nested_layer = layer.layer
  input_shape = layer.input_shape

  # Construct map from attribute (e.g. 'kernel') to tensor versions of
  # variables used during training.
  compressible_training_tensors = {}
  for attr, weights in layer.training_weights.items():
    compressible_training_tensors[attr] = [w.read_value() for w in weights]

  # Process nested layer.
  #
  # TODO(tfmot): same TODOs as in _create_layer_for_training.
  cloned_layer = nested_layer.__class__.from_config(nested_layer.get_config())

  layer_for_inference = _InferenceWrapper(cloned_layer, algorithm,
                                          compressible_training_tensors)
  layer_for_inference.build(input_shape)
  # Clear `_build_input_shape` so that `build()` is not immediately called
  # during reloading. We want the wrapper layer to initiate `build()`.
  layer_for_inference.layer._build_input_shape = None  # pylint: disable=protected-access

  if layer.get_weights():
    # Set weights of layer for inference according to what was trained.
    inference_weights = _map_to_inference_weights(
        layer.get_weights(), algorithm, compressible_training_tensors)
    layer_for_inference.set_weights(inference_weights)

  return layer_for_inference
