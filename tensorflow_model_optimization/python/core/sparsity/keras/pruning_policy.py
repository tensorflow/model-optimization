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
    producers = []
    for node in layer._inbound_nodes:
      if isinstance(node.inbound_layers, list):
        producers.extend(node.inbound_layers)
      else:
        producers.append(node.inbound_layers)
    return producers

  def _get_consumers(self, layer):
    return [node.outbound_layer for node in layer._outbound_nodes]

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
    return (isinstance(layer, layers.Conv2D) and hasattr(layer, 'kernel') and
            layer.kernel.shape[:3] == (3, 3, 3) and layer.strides == (2, 2) and
            layer.padding.lower() == 'valid')

  def _end_layer_stop_fn(self, layer):
    """Determines whether the layer ends a subgraph of sparse inference."""
    return isinstance(layer, layers.GlobalAveragePooling2D) and layer.keepdims

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
    if isinstance(layer, (layers.Add, layers.Multiply, layers.ZeroPadding2D,
                          layers.ReLU, layers.LeakyReLU, layers.ELU)):
      return True
    elif isinstance(layer, layers.DepthwiseConv2D):
      # 3x3 stride-1 convolution (no dilation, padding 1 on each side).
      # 3x3 stride-2 convolution (no dilation, padding 1 on each side).
      # 5x5 stride-1 convolution (no dilation, padding 2 on each side).
      # 5x5 stride-2 convolution (no dilation, padding 2 on each side).
      return (layer.depth_multiplier == 1 and layer.dilation_rate == (1, 1) and
              (layer.kernel_size == (3, 3) or layer.kernel_size == (5, 5)) and
              ((layer.padding.lower() == 'same' and layer.strides == (1, 1)) or
               (layer.padding.lower() == 'valid' and layer.strides == (2, 2))))
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
                        '`input filters == 3` and `VALID` padding in all input '
                        'branches of the model'))

    # Search for the end layer (GlobalAveragePooling with `keepdims = True`)
    # for every output branch (backward).
    output_layers = set(inp._keras_history.layer for inp in model.outputs)
    end_layers = self._lookup_layers(
        output_layers,
        self._end_layer_stop_fn,
        self._get_producers,
    )
    if not end_layers:
      raise ValueError(('Could not find a `GlobalAveragePooling2D` layer with '
                        '`keepdims = True` in all output branches'))

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
