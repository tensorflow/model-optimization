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
"""Entry point for compress models during training."""

import enum
import attr
import tensorflow.compat.v2 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.models import _clone_layers_and_model_config
from tensorflow.python.keras.models import _insert_ancillary_layers
from tensorflow.python.util import nest
# pylint: enable=g-direct-tensorflow-import
from tensorflow_model_optimization.python.core.internal.compression.keras import compress_wrapper


def merge_dicts(*dict_args):
  result = {}
  for dictionary in dict_args:
    result.update(dictionary)
  return result


@attr.s(frozen=True)
class FunctionGroup:
  init = attr.ib()
  training = attr.ib()
  compress = attr.ib()
  decompress = attr.ib()
  kernels = attr.ib()
  num_losses = attr.ib()


class CompressionConfig(object):
  """Compression module config."""

  def __init__(self,
               params=None,
               function_groups=None,
               output_losses=None):
    self.params = params
    self.function_groups = function_groups or []

    self.output_losses = output_losses or []
    self.layers = []
  num_weights = 0

  def add_function_group(
      self, init=None, training=None, compress=None, decompress=None,
      kernels=None, num_losses=0):
    self.function_groups.append(
        FunctionGroup(
            init, training, compress, decompress, kernels, num_losses))

  @classmethod
  def merge(cls, config1, config2):
    """Merge given configs."""
    kwargs = {}
    list_key = [
        'output_losses',
        'function_groups',
    ]
    for key in list_key:
      kwargs[key] = getattr(config1, key) + getattr(config2, key)

    # TODO(kimjaehong): assert(config1.params == config2.params).
    kwargs['params'] = config1.params
    return cls(**kwargs)

  @classmethod
  def build_for_model(cls, model, config):
    raise NotImplementedError()

  def clone_layer(self, layer):
    """clone layer."""
    cloned_layer = layer.__class__.from_config(layer.get_config())

    kernel_vars = []
    for function_group in self.function_groups:
      for klayer, var in function_group.kernels:
        if layer == klayer:
          kernel_vars.append(var)

    if kernel_vars:
      return compress_wrapper.KerasLayerWrapper(cloned_layer, kernel_vars)
    return cloned_layer


def recursive_layer_generator(network):
  for layer in network.layers:
    # print('[recursive_layer_generator] {}'.format(layer))
    yield layer
    if hasattr(layer, 'layers'):
      for sub_layer in recursive_layer_generator(layer):
        yield sub_layer


class LayerVariableWiseCompressionConfig(CompressionConfig):
  """Variable (layer attribute) wise compression config."""

  @classmethod
  def build_for_model(cls, model, params):
    config = cls(params)
    for layer in recursive_layer_generator(model):
      config = cls.merge(config, cls.build_for_layer(layer, params))
    return config

  @classmethod
  def build_for_layer(cls, layer, params):
    config = cls(params)
    weight_keys = params.get_weight_keys(layer.name)
    for weight_key in weight_keys:
      config = cls.merge(
          config,
          cls.build_for_layer_variable(layer, weight_key, params))
    return config

  @classmethod
  def build_for_layer_variable(cls, layer, weight_key, params):
    config = cls(params)

    config.add_function_group(
        config.init, config.training, config.compress, config.decompress,
        [(layer, weight_key)])

    return config

  def init(self, *args):
    return args

  def training(self, *args):
    return self.decompress(*self.compress(*args))

  def compress(self, *args):
    return args

  def decompress(self, *args):
    return args


class CompressionModelPhase(enum.Enum):
  original = 0
  training = 1
  compressed = 2


class WeightLayer(tf.keras.layers.Layer):
  """Weight layer."""

  def __init__(self, spec_dict, **kwargs):
    self.spec_dict = spec_dict
    super(WeightLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    self.weight_dict = self.create_weight_dict(self.spec_dict)
    super(WeightLayer, self).build(input_shape)

  def call(self, inputs=None):
    return self.weight_dict

  def create_weight_dict(self, spec_dict):
    weight_dict = {}
    for key in spec_dict:
      weight_spec = spec_dict[key]
      weight = self.add_weight(**weight_spec.kwargs)
      weight_dict[key] = weight
    return weight_dict

  def assign(self, weight_dict):
    if not self.built:
      self.build(None)

    for key in weight_dict:
      if key in self.weight_dict:
        self.weight_dict[key].assign(weight_dict[key])


def dict_flatten(d):
  keys = nest.flatten(d.keys())
  values = [d[key] for key in keys]
  return keys, values


def wrapped_tf_function_gen(function):
  @tf.function
  def wrapped_tf_function(*inputs):
    return function(*inputs)

  return wrapped_tf_function


def lambda_gen(func, *inputs):
  return lambda: func(*inputs)


class CompressionModel(tf.keras.Model):
  """Compression model."""

  def __init__(
      self,
      config,
      phase,
      name='compression_model',
      training_model=None,
      **kwargs):
    super(CompressionModel, self).__init__(name=name, **kwargs)
    self._config = config
    self._phase = phase
    self.number = 0
    def tensor_like_arg(tensor):
      # pylint: disable=unused-argument
      def initializer(shape, dtype=None):
        return tensor
      # pylint: enable=unused-argument
      self.number += 1
      ret_args = {
          'shape': tensor.shape,
          'dtype': tensor.dtype,
          'initializer': initializer,
          'name': '{}'.format(self.number)
      }
      return ret_args

    if self._phase == CompressionModelPhase.training:
      self.training_weights_list = []
      for function_group in self._config.function_groups:
        training_tensors = function_group.init(*map(
            lambda x: getattr(*x), function_group.kernels))

        training_weights = [*map(
            lambda t: self.add_weight(**tensor_like_arg(t)),
            training_tensors)]
        self.training_weights_list.append(training_weights)

    if self._phase == CompressionModelPhase.compressed:
      self.compressed_weights_list = []
      for idx, function_group in enumerate(self._config.function_groups):
        compressed_tensors = function_group.compress(
            *training_model.training_weights_list[idx])

        compressed_weights = [*map(
            lambda t: self.add_weight(**tensor_like_arg(t)),
            compressed_tensors)]
        self.compressed_weights_list.append(compressed_weights)

  def prevent_constant_folding(self, tensor, dummy_inputs):
    tensor = tf.identity(tensor)
    outputs = tf.cond(tf.reduce_sum(dummy_inputs) > 0,
                      lambda: tensor,
                      lambda: tensor)
    return outputs

  def output_weight_dict_to_output_values(self, output_weight_dict):
    output_weight_spec_keys, _ = dict_flatten(
        self._config.output_weight_spec_dict)
    values = []
    dummy_zero = tf.constant(0.0)
    for key in output_weight_spec_keys:
      if key in output_weight_dict:
        values.append(output_weight_dict[key])
      else:
        values.append(dummy_zero)
    return values

  def call(self, inputs, training=None):
    if self._phase == CompressionModelPhase.training:
      # TODO(kimjaehong): Training-time validation inference.
      if training or True:
        output_weights_list = []
        losses_list = []
        for function_group, training_weights in zip(
            self._config.function_groups, self.training_weights_list):
          training_tensors = [tf.identity(w) for w in training_weights]
          output_weights = function_group.training(*training_tensors)
          n_kernels = len(function_group.kernels)
          output_weights_list += output_weights[:n_kernels]
          losses_list += output_weights[n_kernels:]
        output = []
        output.extend(output_weights_list)
        output.extend(losses_list)
        return output
      else:
        output_weights_list = []
        losses_list = []
        for function_group, training_weights in zip(
            self._config.function_groups, self.training_weights_list):
          training_tensors = [tf.identity(w) for w in training_weights]
          output_weights = function_group.decompress(
              *function_group.compress(*training_tensors))
          n_kernels = len(function_group.kernels)
          output_weights_list += output_weights[:n_kernels]
          losses_list += output_weights[n_kernels:]
        output = []
        output.extend(output_weights_list)
        output.extend(losses_list)
        return output

    elif self._phase == CompressionModelPhase.compressed:
      output_weights_list = []
      losses_list = []
      for function_group, compressed_weights in zip(
          self._config.function_groups, self.compressed_weights_list):
        wrapped_compressed_weights = [self.prevent_constant_folding(w, inputs)
                                      for w in compressed_weights]
        output_weights = function_group.decompress(*wrapped_compressed_weights)
        # output_weights = function_group.decompress(*compressed_weights)
        n_kernels = len(function_group.kernels)
        output_weights_list += output_weights[:n_kernels]
        losses_list += output_weights[n_kernels:]
      output = []
      output.extend(output_weights_list)
      output.extend(losses_list)
      return output

  def get_config(self):
    return {}


def convert_from_model(
    model_orig,
    config,
    phase=CompressionModelPhase.training):
  """Convert a functional `Model` instance.

  Arguments:
      model_orig: Instance of `Model`.
      config: CompressionConfig
      phase: CompressionModelPhase

  Returns:
      An instance of `Model`.

  Raises:
      ValueError: in case of invalid `model` argument value or `layer_fn`
      argument value.
  """
  model_config, created_layers = _clone_layers_and_model_config(
      model_orig, {}, config.clone_layer)

  ############## start hook ##############
  # TODO(kimjaehong): This hook is working for simple model.
  # but need to be generalize.

  dummy_inbound_nodes = [[model_config['input_layers'][0]]]
  # TODO(kimjaehong): Find better name.
  dummy_compression_model_name = 'compression_model'

  model_config['layers'].insert(
      1,
      {
          'inbound_nodes': dummy_inbound_nodes,
          'name': dummy_compression_model_name
      })
  compression_model = CompressionModel(config, phase=phase)
  created_layers[dummy_compression_model_name] = compression_model

  num_losses = 0
  tensor_idx = 0
  for function_group in config.function_groups:
    num_losses += function_group.num_losses
    for layer, weight_key in function_group.kernels:
      for layer_cfg in model_config['layers']:
        if layer_cfg['name'] == layer.name:
          layer_cfg['inbound_nodes'][0][0][3][weight_key] = [
              dummy_compression_model_name, 0, tensor_idx]

      tensor_idx += 1

  output_len = len(model_config['output_layers'])

  for _ in range(num_losses):
    model_config['output_layers'].append(
        [dummy_compression_model_name, 0, tensor_idx])
    tensor_idx += 1
  ############## end hook ##############

  # Reconstruct model from the config, using the cloned layers.
  input_tensors, output_tensors, created_layers = (
      functional.reconstruct_from_config(
          model_config,
          created_layers=created_layers))
  metrics_names = model_orig.metrics_names
  loss_tensors = output_tensors[output_len:]
  output_tensors = output_tensors[:output_len]

  model = tf.keras.Model(input_tensors, output_tensors, name=model_orig.name)
  for loss_tensor in loss_tensors:
    model.add_loss(loss_tensor)

  # Layers not directly tied to outputs of the Model, such as loss layers
  # created in `add_loss` and `add_metric`.
  ancillary_layers = [
      layer for layer in created_layers.values() if layer not in model.layers
  ]
  #### start hook2 ####
  ancillary_layers += config.layers
  #### end hook2 ####
  # pylint: disable=protected-access
  if ancillary_layers:
    new_nodes = nest.flatten([
        layer.inbound_nodes[1:]
        if functional._should_skip_first_node(layer)
        else layer.inbound_nodes for layer in created_layers.values()
    ])
    _insert_ancillary_layers(model, ancillary_layers, metrics_names, new_nodes)
  # pylint: enable=protected-access
  return model


class ConvertHelper(object):
  """Convert helper."""

  def __init__(self, config):
    self.config = config

  def convert_layer_fn(self, layer):
    """Convert layer function."""
    if isinstance(layer, compress_wrapper.ModuleWrapper):
      # TODO(kimjaehong): There are no way to clone custom tf.Module.
      module_wrapper = compress_wrapper.ModuleWrapper(
          layer.constructor, layer.var_names, layer.setter)
      module_wrapper.build()
      return module_wrapper

    if isinstance(layer, compress_wrapper.KerasLayerWrapper):
      sub_layer = layer.layer
      cloned_sub_layer = sub_layer.__class__.from_config(sub_layer.get_config())
      return compress_wrapper.KerasLayerWrapper(cloned_sub_layer, ['kernel'])

    if isinstance(layer, CompressionModel):
      self.before = layer
      self.after = CompressionModel(
          self.config, training_model=layer,
          phase=CompressionModelPhase.compressed)
      return self.after

    cloned_layer = layer.__class__.from_config(layer.get_config())
    return cloned_layer


def convert_to_compressed_phase_from_training_phase(model_training, config):
  """Convert to compression phase from training phase model."""
  helper = ConvertHelper(config)
  model_config, created_layers = _clone_layers_and_model_config(
      model_training, {}, helper.convert_layer_fn)

  # Reconstruct model from the config, using the cloned layers.
  input_tensors, output_tensors, created_layers = (
      functional.reconstruct_from_config(
          model_config,
          created_layers=created_layers))
  metrics_names = model_training.metrics_names
  model = tf.keras.Model(
      input_tensors, output_tensors, name=model_training.name)
  # Layers not directly tied to outputs of the Model, such as loss layers
  # created in `add_loss` and `add_metric`.
  ancillary_layers = [
      layer for layer in created_layers.values() if layer not in model.layers
  ]
  #### start hook ####
  ancillary_layers += config.layers
  #### end hook ####
  # pylint: disable=protected-access
  if ancillary_layers:
    new_nodes = nest.flatten([
        layer.inbound_nodes[1:]
        if functional._should_skip_first_node(layer)
        else layer.inbound_nodes for layer in created_layers.values()
    ])
    _insert_ancillary_layers(model, ancillary_layers, metrics_names, new_nodes)
  # pylint: enable=protected-access
  #
  # TODO(kimjaehong): set weight model_training => model.
  #
  ######## start hook2 ########

  orig_layers = {layer.name: layer for layer in model_training.layers}
  for key in orig_layers:
    layer = orig_layers[key]
    if isinstance(layer, CompressionModel):
      continue
    created_layers[key].set_weights(layer.get_weights())

  ######## end hook2 ########

  return model


class DecompressHelper(object):
  """Decompress helper."""

  def __init__(self, config):
    self.config = config

  def convert_layer_fn(self, layer):
    """Convert layer function."""
    if isinstance(layer, compress_wrapper.ModuleWrapper):
      # TODO(kimjaehong): There are no way to clone custom tf.Module.
      return None

    if isinstance(layer, compress_wrapper.KerasLayerWrapper):
      sub_layer = layer.layer
      cloned_sub_layer = sub_layer.__class__.from_config(sub_layer.get_config())
      return cloned_sub_layer

    if isinstance(layer, CompressionModel):
      self.compression_model = layer
      return None

    cloned_layer = layer.__class__.from_config(layer.get_config())
    return cloned_layer


def remove_none_inbound_nodes(inbound_nodes, created_layers):
  """remove none inbound nodes during conversion."""
  new_inbound_nodes = []
  num_removed = 0
  for inbound_node in inbound_nodes:
    if created_layers[inbound_node[0][0]]:
      inbound_node_dict = inbound_node[0][3]
      new_inbound_node_dict = {}
      for key in inbound_node_dict:
        if created_layers[inbound_node_dict[key][0]]:
          new_inbound_node_dict[key] = inbound_node_dict[key]
        else:
          num_removed += 1
      new_inbound_node = [[
          inbound_node[0][0],
          inbound_node[0][1],
          inbound_node[0][2],
          new_inbound_node_dict
      ]]
      new_inbound_nodes.append(new_inbound_node)
    else:
      num_removed += 1

  return new_inbound_nodes, num_removed


def remove_none(layers, created_layers):
  """remove none."""
  new_layers = []
  new_created_layers = {}

  for key in created_layers:
    if created_layers[key]:
      new_created_layers[key] = created_layers[key]

  for layer in layers:
    if created_layers[layer['name']]:
      new_layer = dict(layer)
      new_layer['inbound_nodes'], num_removed = remove_none_inbound_nodes(
          layer['inbound_nodes'], created_layers)

      # TODO(kimjaehong): find more accurate way to remove extra loss.
      if num_removed > 0 and layer['name'].startswith('add_loss'):
        del new_created_layers[layer['name']]
        continue

      new_layers.append(new_layer)

  return new_layers, new_created_layers


def convert_to_original_from_compressed_phase(compressed_model, config):
  """Convert to compression phase from training phase model."""
  helper = DecompressHelper(config)
  model_config, created_layers = _clone_layers_and_model_config(
      compressed_model, {}, helper.convert_layer_fn)

  model_config['layers'], created_layers = \
    remove_none(model_config['layers'], created_layers)
  # Reconstruct model from the config, using the cloned layers.
  input_tensors, output_tensors, created_layers = (
      functional.reconstruct_from_config(
          model_config,
          created_layers=created_layers))
  metrics_names = compressed_model.metrics_names
  model = tf.keras.Model(
      input_tensors, output_tensors, name=compressed_model.name)
  # Layers not directly tied to outputs of the Model, such as loss layers
  # created in `add_loss` and `add_metric`.
  ancillary_layers = [
      layer for layer in created_layers.values() if layer not in model.layers
  ]
  # pylint: disable=protected-access
  if ancillary_layers:
    new_nodes = nest.flatten([
        layer.inbound_nodes[1:]
        if functional._should_skip_first_node(layer)
        else layer.inbound_nodes for layer in created_layers.values()
    ])
    _insert_ancillary_layers(model, ancillary_layers, metrics_names, new_nodes)
  # pylint: enable=protected-access
  #
  # TODO(kimjaehong): set weight model_training => model.
  #
  ######## start hook ########

  # TODO(kimjaehong): currently only support conv / dense layer for kernel.
  # pytype: disable=attribute-error
  output_weights = helper.compression_model(tf.constant(0.))
  output_weight_map = config.output_weight_map
  output_weight_spec_keys, _ = dict_flatten(config.output_weight_spec_dict)

  orig_layers = {layer.name: layer for layer in compressed_model.layers}
  for key in orig_layers:
    layer = orig_layers[key]
    if isinstance(layer, CompressionModel):
      continue
    if key in created_layers:
      weights = layer.get_weights()
      to_layer = created_layers[key]
      if to_layer.name in output_weight_map:
        prepend_weights = []
        for weight_key in output_weight_map[to_layer.name]:
          tensor_idx = output_weight_spec_keys.index(
              output_weight_map[to_layer.name][weight_key])
          prepend_weights.append(output_weights[tensor_idx])
        # TODO(kimjaehong): Should it be numpy array?
        weights = prepend_weights + weights
      to_layer.set_weights(weights)
  # pytype: enable=attribute-error

  ######## end hook ########

  return model


class Parameters(object):

  def __init__(self):
    pass

  def get_config(self):
    return {}


class LayerVariableWiseParameters(Parameters):
  """Configuration for layer-wise compression algorithm."""

  def __init__(self, layer_name_to_weight_keys_map=None):
    super().__init__()
    self.layer_name_to_weight_keys_map = layer_name_to_weight_keys_map

  def get_weight_keys(self, layer_name):
    if layer_name in self.layer_name_to_weight_keys_map:
      return self.layer_name_to_weight_keys_map[layer_name]
    return []

  def get_config(self):
    return {
        'layer_name_to_weight_keys_map': self.layer_name_to_weight_keys_map,
    }

