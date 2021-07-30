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
# pylint: disable=protected-access
"""Pruning Policy classes to control application of pruning wrapper."""

import abc
import tensorflow as tf

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

layers = tf.keras.layers
activations = tf.keras.activations


class PruningPolicy(abc.ABC):
  """Specifies what layers to prune in the model.

  PruningPolicy controls application of `PruneLowMagnitude` wrapper on per-layer
  basis and checks that the model contains only supported layers.
  PruningPolicy works together with `prune_low_magnitude` through which it
  provides fine-grained control over pruning in the model.

  ```python
  pruning_params = {
      'pruning_schedule': ConstantSparsity(0.5, 0),
      'block_size': (1, 1),
      'block_pooling_type': 'AVG'
  }

  model = prune_low_magnitude(
      keras.Sequential([
          layers.Dense(10, activation='relu', input_shape=(100,)),
          layers.Dense(2, activation='sigmoid')
      ]),
      pruning_policy=PruneForLatencyOnXNNPack(),
      **pruning_params)
  ```

  You can inherit this class to write your own custom pruning policy.

  The API is experimental and is subject to change.
  """

  @abc.abstractmethod
  def allow_pruning(self, layer):
    """Checks if pruning wrapper should be applied for the current layer.

    Args:
      layer: Current layer in the model.

    Returns:
      True/False, whether the pruning wrapper should be applied for the layer.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def ensure_model_supports_pruning(self, model):
    """Checks that the model contains only supported layers.

    Args:
      model: A `tf.keras.Model` instance which is going to be pruned.

    Raises:
      ValueError: if the keras model doesn't support pruning policy, i.e. keras
        model contains an unsupported layer.
    """
    raise NotImplementedError


class PruneForLatencyOnXNNPack(PruningPolicy):
  """Specifies to prune only 1x1 Conv2D layers in the model.

  PruneForLatencyOnXNNPack checks that the model contains a subgraph that can
  leverage XNNPACK's sparse inference and applies pruning wrapper only to
  Conv2D with `kernel_size = (1, 1)`.

  Reference:
    - [Fast Sparse ConvNets](https://arxiv.org/abs/1911.09723)
    - [XNNPACK Sparse Inference](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md#sparse-inference) # pylint: disable=line-too-long
  """

  def allow_pruning(self, layer):
    """Allows to prune only 1x1 Conv2D layers."""
    return isinstance(layer, layers.Conv2D) and layer.kernel_size == (1, 1)

  def _get_producers(self, layer):
    producers = set()
    for node in layer._inbound_nodes:
      if isinstance(node.inbound_layers, list):
        producers.update(node.inbound_layers)
      else:
        producers.add(node.inbound_layers)
    return producers

  def _get_consumers(self, layer):

    def unpack(layer):
      return (unpack(layer.layers[0])
              if isinstance(layer, tf.keras.Sequential) else layer)

    return [unpack(node.outbound_layer) for node in layer._outbound_nodes]

  def _lookup_layers(self, source_layers, stop_fn, next_fn):
    """Traverses the model and returns layers satisfying `stop_fn` criteria."""
    to_visit = set(source_layers)
    used_layers = set(source_layers)
    found_layers = set()
    while to_visit:
      layer = to_visit.pop()
      if stop_fn(layer):
        found_layers.add(layer)
      else:
        next_layers = next_fn(layer)
        if not next_layers:
          return set()
        for next_layer in next_layers:
          if next_layer not in used_layers:
            used_layers.add(next_layer)
            to_visit.add(next_layer)

    return found_layers

  def _start_layer_stop_fn(self, layer):
    """Determines whether the layer starts a subgraph of sparse inference."""
    if isinstance(layer, layers.Conv2D):
      producers = list(self._get_producers(layer))
      return (hasattr(layer, 'kernel') and
              layer.kernel.shape[:3] == (3, 3, 3) and
              layer.strides == (2, 2) and layer.padding.lower() == 'valid' and
              len(producers) == 1 and
              isinstance(producers[0], layers.ZeroPadding2D) and
              producers[0].padding == ((1, 1), (1, 1)))
    return False

  def _end_layer_stop_fn(self, layer):
    """Determines whether the layer ends a subgraph of sparse inference."""
    return isinstance(layer, layers.GlobalAveragePooling2D)

  def _check_layer_support(self, layer):
    """Returns whether the layer is supported or not.

    Mimics XNNPACK's behaviour of compatibility function.

    Args:
      layer: Current layer in the model.

    Returns:
      True if the layer is supported, False otherwise.

    References:
      - https://github.com/google/XNNPACK/blob/master/src/subgraph.c#L130
    """
    if isinstance(layer,
                  (layers.Add, layers.Multiply, layers.ZeroPadding2D,
                   layers.ReLU, layers.LeakyReLU, layers.ELU, layers.Dropout)):
      return True
    elif isinstance(layer, layers.DepthwiseConv2D):
      # 3x3 convolution with `SAME` padding (no dilation, stride-1).
      # 3x3 convolution with `VALID` padding (no dilation, stride-1 or stride-2,
      #   preceding `ZeroPadding2D` layer with padding 1 on each side.
      # 5x5 convolution with `SAME` padding (no dilation, stride-1)
      # 5x5 convolution with `VALID` padding (no dilation, stride-1 or stride-2,
      #   preceding `ZeroPadding2D` layer with padding 2 on each side.
      padding = layer.padding.lower()
      producers = list(self._get_producers(layer))
      zero_padding = (
          producers[0] if len(producers) == 1 and
          isinstance(producers[0], layers.ZeroPadding2D) else None)

      supported_case_1 = (
          layer.kernel_size == (3, 3) and layer.strides == (1, 1) and
          padding == 'same')

      supported_case_2 = (
          layer.kernel_size == (3, 3) and
          (layer.strides == (1, 1) or layer.strides == (2, 2)) and
          padding == 'valid' and zero_padding and
          zero_padding.padding == ((1, 1), (1, 1)))

      supported_case_3 = (
          layer.kernel_size == (5, 5) and layer.strides == (1, 1) and
          padding == 'same')

      supported_case_4 = (
          layer.kernel_size == (5, 5) and
          (layer.strides == (1, 1) or layer.strides == (2, 2)) and
          padding == 'valid' and zero_padding and
          zero_padding.padding == ((2, 2), (2, 2)))

      supported = (
          layer.depth_multiplier == 1 and layer.dilation_rate == (1, 1) and
          (supported_case_1 or supported_case_2 or supported_case_3 or
           supported_case_4))

      return supported
    elif isinstance(layer, layers.Conv2D):
      # 1x1 convolution (no stride, no dilation, no padding, no groups).
      return (layer.groups == 1 and layer.dilation_rate == (1, 1) and
              layer.kernel_size == (1, 1) and layer.strides == (1, 1))
    elif isinstance(layer, layers.GlobalAveragePooling2D):
      return layer.keepdims
    elif isinstance(layer, layers.BatchNormalization):
      return list(layer.axis) == [3]
    elif isinstance(layer, layers.UpSampling2D):
      return layer.interpolation == 'bilinear'
    elif isinstance(layer, layers.Activation):
      return activations.serialize(layer.activation) in ('relu', 'relu6',
                                                         'leaky_relu', 'elu',
                                                         'sigmoid')
    elif layer.__class__.__name__ == 'TFOpLambda':
      return layer.function in (tf.identity, tf.__operators__.add, tf.math.add,
                                tf.math.subtract, tf.math.multiply)
    elif isinstance(layer, pruning_wrapper.PruneLowMagnitude):
      return self._check_layer_support(layer.layer)
    return False

  def ensure_model_supports_pruning(self, model):
    """Ensures that the model contains only supported layers."""

    # Check whether the model is a subclass model.
    if (not model._is_graph_network and
        not isinstance(model, tf.keras.models.Sequential)):
      raise ValueError('Subclassed models are not supported currently.')

    if not model.built:
      raise ValueError('Unbuilt models are not supported currently.')

    # Gather the layers that consume model's input tensors.
    input_layers = set(inp._keras_history.layer for inp in model.inputs)

    # Search for the start layer (Conv2D 3x3, `stride = (2, 2)`,
    # `filters = 3`, `padding = `VALID``) in every input branch (forward).
    start_layers = self._lookup_layers(
        input_layers,
        self._start_layer_stop_fn,
        self._get_consumers,
    )
    if not start_layers:
      raise ValueError(('Could not find `Conv2D 3x3` layer with stride 2x2, '
                        '`input filters == 3` and `VALID` padding and '
                        'preceding `ZeroPadding2D` with `padding == 1` in all '
                        'input branches of the model'))

    # Search for the end layer (GlobalAveragePooling)
    # for every output branch (backward).
    output_layers = set(inp._keras_history.layer for inp in model.outputs)
    end_layers = self._lookup_layers(
        output_layers,
        self._end_layer_stop_fn,
        self._get_producers,
    )
    if not end_layers:
      raise ValueError(('Could not find a `GlobalAveragePooling2D` layer in '
                        'all output branches'))

    # Ensure that all layers between the start and the end layers are supported
    # for pruning.
    def visit_fn(layer):
      if layer not in end_layers and not self._check_layer_support(layer):
        raise ValueError(('Layer {layer} is not supported for the {policy} '
                          'pruning policy'.format(
                              layer=layer.__class__.__name__,
                              policy=self.__class__.__name__)))
      return layer in end_layers

    _ = self._lookup_layers(
        sum([self._get_consumers(layer) for layer in start_layers], []),
        visit_fn,
        self._get_consumers,
    )
