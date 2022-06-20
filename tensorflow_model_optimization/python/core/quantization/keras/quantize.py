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
"""Quantization API functions for tf.keras models."""
import warnings

import tensorflow as tf

from tensorflow_model_optimization.python.core.keras import metrics
from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate as quantize_annotate_mod
from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config as quantize_config_mod
from tensorflow_model_optimization.python.core.quantization.keras import quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras import quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_scheme
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit import default_n_bit_quantize_registry

keras = tf.keras


def quantize_scope(*args):
  """Scope which can be used to deserialize quantized Keras models and layers.

  Under `quantize_scope`, Keras methods such as `tf.keras.load_model` and
  `tf.keras.models.model_from_config` will be able to deserialize Keras models
  and layers which contain quantization classes such as `QuantizeConfig`
  and `Quantizer`.

  Example:

  ```python
  tf.keras.models.save_model(quantized_model, keras_file)

  with quantize_scope():
    loaded_model = tf.keras.models.load_model(keras_file)

  # If your quantized model uses custom objects such as a specific `Quantizer`,
  # you can pass them to quantize_scope to deserialize your model.
  with quantize_scope({'FixedRangeQuantizer', FixedRangeQuantizer}
    loaded_model = tf.keras.models.load_model(keras_file)
  ```

  For further understanding, see `tf.keras.utils.custom_object_scope`.

  Args:
    *args: Variable length list of dictionaries of `{name, class}` pairs to add
      to the scope created by this method.

  Returns:
    Object of type `CustomObjectScope` with quantization objects included.
  """
  quantization_objects = {
      'QuantizeAnnotate': quantize_annotate_mod.QuantizeAnnotate,
      'QuantizeAwareActivation':
          quantize_aware_activation.QuantizeAwareActivation,
      'NoOpActivation': quantize_aware_activation.NoOpActivation,
      'QuantizeWrapper': quantize_wrapper.QuantizeWrapper,
      'QuantizeWrapperV2': quantize_wrapper.QuantizeWrapperV2,
      'QuantizeLayer': quantize_layer.QuantizeLayer,
      'OutputOnlyConfig': quantize_config_mod.OutputOnlyConfig,
      'FixedQuantizeConfig': quantize_config_mod.FixedQuantizeConfig,
  }
  quantization_objects.update(default_8bit_quantize_registry._types_dict())  # pylint: disable=protected-access
  quantization_objects.update(default_n_bit_quantize_registry._types_dict())  # pylint: disable=protected-access
  quantization_objects.update(quantizers._types_dict())  # pylint: disable=protected-access

  return tf.keras.utils.custom_object_scope(*(args + (quantization_objects,)))


def quantize_model(to_quantize):
  """Quantize a `tf.keras` model with the default quantization implementation.

  Quantization constructs a model which emulates quantization during training.
  This allows the model to learn parameters robust to quantization loss, and
  also model the accuracy of a quantized model.

  For more information, see
  https://www.tensorflow.org/model_optimization/guide/quantization/training

  Quantize a model:

  ```python
  # Quantize sequential model
  model = quantize_model(
      keras.Sequential([
          layers.Dense(10, activation='relu', input_shape=(100,)),
          layers.Dense(2, activation='sigmoid')
      ]))

  # Quantize functional model
  in = tf.keras.Input((3,))
  out = tf.keras.Dense(2)(in)
  model = tf.keras.Model(in, out)

  quantized_model = quantize_model(model)
  ```

  Note that this function removes the optimizer from the original model.

  The returned model copies over weights from the original model. So while
  it preserves the original weights, training it will not modify the weights
  of the original model.

  Args:
    to_quantize: tf.keras model to be quantized. It can have pre-trained
      weights.

  Returns:
    Returns a new `tf.keras` model prepared for quantization.
  """
  if to_quantize is None:
    raise ValueError('`to_quantize` cannot be None')

  if not isinstance(to_quantize, keras.Model):
    raise ValueError(
        '`to_quantize` can only be a `tf.keras.Model` instance. Use '
        'the `quantize_annotate_layer` API to handle individual layers.'
        'You passed an instance of type: {input}.'.format(
            input=to_quantize.__class__.__name__))

  if not isinstance(
      to_quantize, keras.Sequential) and not to_quantize._is_graph_network:  # pylint: disable=protected-access
    raise ValueError(
        '`to_quantize` can only either be a tf.keras Sequential or '
        'Functional model.')

  annotated_model = quantize_annotate_model(to_quantize)
  return quantize_apply(annotated_model)


def quantize_annotate_model(to_annotate):
  """Annotate a `tf.keras` model to be quantized.

  This function does not actually quantize the model. It merely specifies
  that the model needs to be quantized. `quantize_apply` can then be used
  to quantize the model.

  This function is intended to be used in conjunction with the
  `quantize_annotate_layer` API. Otherwise, it is simpler to use
  `quantize_model`.

  Annotate a model while overriding the default behavior for a layer:

  ```python
  quantize_config = MyDenseQuantizeConfig()

  model = quantize_annotate_model(
    keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      quantize_annotate_layer(
          layers.Dense(2, activation='sigmoid'),
          quantize_config=quantize_config)
    ]))

  # The first Dense layer gets quantized with the default behavior,
  # but the second layer uses `MyDenseQuantizeConfig` for quantization.
  quantized_model = quantize_apply(model)
  ```

  Note that this function removes the optimizer from the original model.

  Args:
    to_annotate: `tf.keras` model which needs to be quantized.

  Returns:
    New tf.keras model with each layer in the model wrapped with
    `QuantizeAnnotate`. The new model preserves weights from the original
    model.

  Raises:
    ValueError: if the model cannot be annotated.
  """
  if to_annotate is None:
    raise ValueError('`to_annotate` cannot be None')

  if not isinstance(to_annotate, keras.Model):
    raise ValueError(
        '`to_annotate` can only be a `tf.keras.Model` instance. Use '
        'the `quantize_annotate_layer` API to handle individual layers. '
        'You passed an instance of type: {input}.'.format(
            input=to_annotate.__class__.__name__))

  if not isinstance(
      to_annotate, keras.Sequential) and not to_annotate._is_graph_network:  # pylint: disable=protected-access
    raise ValueError(
        '`to_annotate` can only either be a tf.keras Sequential or '
        'Functional model.')

  def _add_quant_wrapper(layer):
    """Add annotation wrapper."""
    # Already annotated layer. No need to wrap.
    if isinstance(layer, quantize_annotate_mod.QuantizeAnnotate):
      return layer

    if isinstance(layer, tf.keras.layers.Lambda):
      warnings.warn(
          'Lambda layers are not supported by automatic model annotation '
          'because the internal functionality cannot always be determined by '
          'serialization alone. We recommend that you make a custom layer '
          'and add a custom QuantizeConfig for it instead. This layer will not '
          'be quantized which may lead to unexpected results.')
      return layer

    if isinstance(layer, tf.keras.Model):
      raise ValueError(
          'Quantizing a tf.keras Model inside another tf.keras Model is not supported.'
      )

    return quantize_annotate_mod.QuantizeAnnotate(layer)

  return keras.models.clone_model(
      to_annotate, input_tensors=None, clone_function=_add_quant_wrapper)


def quantize_annotate_layer(to_annotate, quantize_config=None):
  """Annotate a `tf.keras` layer to be quantized.

  This function does not actually quantize the layer. It is merely used to
  specify that the layer should be quantized. The layer then gets quantized
  accordingly when `quantize_apply` is used.

  This method should be used when the user wants to quantize only certain
  layers of the model, or change the default behavior of how a layer is
  quantized.

  Annotate a layer:

  ```python
  model = keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      quantize_annotate_layer(layers.Dense(2, activation='sigmoid'))
  ])

  # Only the second Dense layer is quantized.
  quantized_model = quantize_apply(model)
  ```

  Args:
    to_annotate: `tf.keras` layer which needs to be quantized.
    quantize_config: optional `QuantizeConfig` which controls how the layer is
      quantized. In its absence, the default behavior for the layer is used.

  Returns:
    `tf.keras` layer wrapped with `QuantizeAnnotate`.
  """
  if to_annotate is None:
    raise ValueError('`to_annotate` cannot be None')

  # Check against keras.Model since it is an instance of keras.layers.Layer.
  if not isinstance(to_annotate, keras.layers.Layer) or isinstance(
      to_annotate, keras.Model):
    raise ValueError(
        '`to_annotate` can only be a `tf.keras.layers.Layer` instance. '
        'You passed an instance of type: {input}.'.format(
            input=to_annotate.__class__.__name__))

  if quantize_config is not None and not isinstance(
      quantize_config, quantize_config_mod.QuantizeConfig):
    raise ValueError(
        '`quantize_config` can only be a `tfmot.quantization.keras.QuantizeConfig` instance.'
        'You passed an instance of type: {input}.'.format(
            input=quantize_config.__class__.__name__))

  return quantize_annotate_mod.QuantizeAnnotate(
      layer=to_annotate, quantize_config=quantize_config)


@metrics.MonitorBoolGauge('quantize_apply_usage')
def quantize_apply(
    model,
    scheme=default_8bit_quantize_scheme.Default8BitQuantizeScheme()):
  """Quantize a `tf.keras` model that has been annotated for quantization.

  Quantization constructs a model which emulates quantization during training.
  This allows the model to learn parameters robust to quantization loss, and
  also model the accuracy of a quantized model.

  For more information, see
  https://www.tensorflow.org/model_optimization/guide/quantization/training
  TODO(tfmot): Link blog once launched.

  This function takes a `tf.keras` model in which the desired layers for
  quantization have already been annotated. See `quantize_annotate_model`
  and `quantize_annotate_layer`.

  Example:

  ```python
  model = keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      quantize_annotate_layer(layers.Dense(2, activation='sigmoid'))
  ])

  # Only the second Dense layer is quantized.
  quantized_model = quantize_apply(model)
  ```

  Note that this function removes the optimizer from the original model.

  The returned model copies over weights from the original model. So while
  it preserves the original weights, training it will not modify the weights
  of the original model.

  Args:
    model: A `tf.keras` Sequential or Functional model which has been annotated
      with `quantize_annotate`. It can have pre-trained weights.
    scheme: A `QuantizeScheme` which specifies transformer and quantization
      registry. The default is `Default8BitQuantizeScheme()`.

  Returns:
    Returns a new `tf.keras` model in which the annotated layers have been
    prepared for quantization.
  """
  if model is None:
    raise ValueError('`model` cannot be None')

  if not isinstance(model, keras.Model):
    raise ValueError('`model` can only be a `tf.keras.Model` instance.'
                     'You passed an instance of type: {input}.'.format(
                         input=model.__class__.__name__))

  if not isinstance(model, keras.Sequential) and not model._is_graph_network:  # pylint: disable=protected-access
    raise ValueError('`model` can only either be a tf.keras Sequential or '
                     'Functional model.')

  # Have at least 1 layer annotated with QuantizeAnnotate
  if not any(isinstance(layer, quantize_annotate_mod.QuantizeAnnotate)
             for layer in model.layers):
    raise ValueError('`model` must contain at least one layer which have been '
                     'annotated with `quantize_annotate*`. There are no layers '
                     'to quantize.')

  if not model.built:
    raise ValueError('`model` must be a built model. '
                     'been built yet. Please call `model.build(input_shape)` '
                     'before quantizing your model.')

  def _clone_model_with_weights(model_to_clone):
    cloned_model = keras.models.clone_model(model_to_clone)
    cloned_model.set_weights(model_to_clone.get_weights())

    return cloned_model

  def _extract_original_model(model_to_unwrap):
    """Extracts original model by removing wrappers."""
    layer_quantize_map = {}
    requires_output_quantize = set()

    def _unwrap(layer):
      if not isinstance(layer, quantize_annotate_mod.QuantizeAnnotate):
        return layer

      annotate_wrapper = layer
      # pylint: disable=protected-access
      if layer._inbound_nodes and len(layer._inbound_nodes) == 1:
        node = layer._inbound_nodes[0]
        inbound_layers = tf.nest.flatten(node.inbound_layers)
        if len(inbound_layers) == 1 and not isinstance(
            inbound_layers[0], quantize_annotate_mod.QuantizeAnnotate):
          requires_output_quantize.add(inbound_layers[0].name)
      # pylint: enable=protected-access

      layer_quantize_map[annotate_wrapper.layer.name] = {
          'quantize_config': annotate_wrapper.quantize_config
      }
      return annotate_wrapper.layer

    unwrapped_model = keras.models.clone_model(
        model_to_unwrap, input_tensors=None, clone_function=_unwrap)
    return unwrapped_model, layer_quantize_map, requires_output_quantize

  def _quantize(layer):  # pylint: disable=missing-docstring
    # Handle quantize layer before any layers.
    # layer is a QuantizeLayer, possibly rebuild
    # layer with modified config from parameters stored in the map.
    if isinstance(layer, quantize_layer.QuantizeLayer):
      # If there is more than one usage of the input, even if all are concat,
      # we need to quantize.
      if len(layer._outbound_nodes) > 1:  # pylint: disable=protected-access
        return layer

      layer_config = layer.get_config()
      if layer.name not in layer_quantize_map:  # Possibly added manually.
        with quantize_scope():
          return quantize_layer.QuantizeLayer.from_config(layer_config)

      for key, value in layer_quantize_map[layer.name].items():
        layer_config[key] = value
      return quantize_layer.QuantizeLayer.from_config(layer_config)

    if ((layer.name not in layer_quantize_map and
         layer.name not in requires_output_quantize) or
        (isinstance(layer, quantize_wrapper.QuantizeWrapper))):
      # It supports for custom QuantizeWrapper.
      return layer

    if layer.name in requires_output_quantize:
      if not quantize_registry.supports(layer):
        return layer
      full_quantize_config = quantize_registry.get_quantize_config(layer)
      if not full_quantize_config:
        return layer
      quantize_config = quantize_config_mod.OutputOnlyConfig(
          full_quantize_config)
    else:
      quantize_config = layer_quantize_map[layer.name].get('quantize_config')
      if not quantize_config and quantize_registry.supports(layer):
        quantize_config = quantize_registry.get_quantize_config(layer)

    if not quantize_config:
      error_msg = (
          'Layer {}:{} is not supported. You can quantize this '
          'layer by passing a `tfmot.quantization.keras.QuantizeConfig` '
          'instance to the `quantize_annotate_layer` '
          'API.')
      raise RuntimeError(
          error_msg.format(layer.name, layer.__class__,
                           quantize_registry.__class__))

    # `QuantizeWrapper` does not copy any additional layer params from
    # `QuantizeAnnotate`. This should generally be fine, but occasionally
    # `QuantizeAnnotate` wrapper may contain `batch_input_shape` like params.
    # TODO(pulkitb): Ensure this does not affect model cloning.
    return quantize_wrapper.QuantizeWrapperV2(
        layer, quantize_config)

  # 1. Create a copy of the model with the same weights. This ensures
  # modifications don't affect the original model, or its weights.
  try:
    model_copy = _clone_model_with_weights(model)
  except ValueError as er:
    raise ValueError(
        'Unable to clone model. This generally happens if you used custom '
        'Keras layers or objects in your model. Please specify them via '
        '`quantize_scope` for your calls to `quantize_model` and '
        '`quantize_apply`. [%s].' % er)

  # 2. Remove QuantizeAnnotate wrappers from the layers in the model. This
  # extracts the original model structure (easier to transform), and
  # stores relevant quantization information in a map.
  (unwrapped_model, layer_quantize_map,
   requires_output_quantize) = _extract_original_model(model_copy)
  # Model cloning excludes input layers. Add input layers into the map
  # since they need to be matched for patterns as well.
  # pylint: disable=protected-access
  for input_layer in unwrapped_model._input_layers:
    for outbound_node in input_layer._outbound_nodes:
      if outbound_node.outbound_layer.name in layer_quantize_map:
        layer_quantize_map[input_layer.name] = {}
  # pylint: enable=protected-access

  # 3. Apply the graph transformations required to match model passes on
  # target device/dialect.
  quantize_transform = scheme.get_layout_transformer()
  # layer_quantize_map gets modified by the transformations.
  transformed_model, layer_quantize_map = quantize_transform.apply(
      unwrapped_model, layer_quantize_map)

  # TODO(pulkitb): Think more about how to introduce Default specific code.
  quantize_registry = scheme.get_quantize_registry()

  # 4. Actually quantize all the relevant layers in the model. This is done by
  # wrapping the layers with QuantizeWrapper, and passing the associated
  # `QuantizeConfig`.

  return keras.models.clone_model(
      transformed_model, input_tensors=None, clone_function=_quantize)


def _unwrap_first_input_name(inbound_nodes):
  """Unwrap inbound_nodes three times to get first input name.

  Args:
    inbound_nodes: A str config that indicates input node. This method assumed
      the inbound_nodes looks like `[[['input', 0, 0, {}]]]`.

  Returns:
    Returns a str name for the first inbound node.
  """
  current = inbound_nodes

  for _ in range(3):
    if not current:
      return None
    if not isinstance(current, list):
      return None
    current = current[0]

  if isinstance(current, str):
    return current

  return None


def _wrap_fixed_range(
    quantize_config, num_bits, init_min, init_max, narrow_range):
  config = quantize_config_mod.FixedQuantizeConfig.from_config(
      {'config': quantize_config,
       'num_bits': num_bits,
       'init_min': init_min,
       'init_max': init_max,
       'narrow_range': narrow_range})
  return tf.keras.utils.serialize_keras_object(config)


def _is_serialized_node_data(nested):
  # Node data can be of form `[layer_name, node_id, tensor_id]` or
  # `[layer_name, node_id, tensor_id, kwargs]`.
  if (isinstance(nested, list) and (len(nested) in [3, 4]) and
      isinstance(nested[0], str)):
    return True
  return False


def _nested_to_flatten_node_data_list(nested):
  """Makes nested node data to flatten node data list."""
  if _is_serialized_node_data(nested):
    return [nested]

  if isinstance(nested, list):
    return sum(map(_nested_to_flatten_node_data_list, nested), [])

  if isinstance(nested, dict):
    return sum(map(_nested_to_flatten_node_data_list, nested.values()), [])

  raise ValueError('{} is not a supported nested node data.'.format(nested))


def fix_input_output_range(
    model,
    num_bits=8,
    input_min=0.0,
    input_max=1.0,
    output_min=0.0,
    output_max=1.0,
    narrow_range=False):
  """Fix the input and output ranges.

  Example:

  ```python
  model = keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      quantize_annotate_layer(layers.Dense(2, activation='sigmoid'))
  ])
  with quantize.quantize_scope():
    model = quantize_annotate_model(model)
    model = quantize_apply(model)
    model = fix_input_output_range(model, num_bits=4,
        input_min=0, input_max=15,
        output_min=0, output_max=15,
        narrow_range=False)
  ```

  In certain cases, a desired input/output ranges is known and should not be
  altered during training. To set these values, use the arguments as follows:

  Args:
    model: A `tf.keras` Sequential or Functional model which has been quantized.
    num_bits: Number of bits for quantization
    input_min: The lower end of quantization interval for the input.
    input_max: The upper end of quantization interval for the input.
    output_min: The lower end of quantization interval for the output.
    output_max: The upper end of quantization interval for the output.
    narrow_range: In case of 8 bits, narrow_range nudges the quantized range
      to be [-127, 127] instead of [-128, 127]. This ensures symmetric
      range has 0 as the centre.

  Returns:
    Returns a new `tf.keras` model fixed input range set to (input_min,
    input_max) and fixed output range set to (output_min, output_max).
  """
  config = model.get_config()
  fixed_input_quantizer = quantizers.FixedQuantizer(
      num_bits=num_bits,
      init_min=input_min,
      init_max=input_max,
      narrow_range=narrow_range)
  serialized_fixed_input_quantizer = tf.keras.utils.serialize_keras_object(
      fixed_input_quantizer)

  if _is_functional_model(model):
    input_layer_list = _nested_to_flatten_node_data_list(config['input_layers'])
    for layer_config in config['layers']:
      input_name = _unwrap_first_input_name(layer_config['inbound_nodes'])
      if input_name is None:
        continue

      for input_layer in input_layer_list:
        if input_name == input_layer[0]:
          layer_config['config']['quantizer'] = serialized_fixed_input_quantizer
          break

    output_layer_list = _nested_to_flatten_node_data_list(
        config['output_layers'])
    for layer_config in config['layers']:
      for output_layer in output_layer_list:
        if layer_config['config']['name'] == output_layer[0]:
          if 'quantize_config' in layer_config['config']:
            layer_config['config']['quantize_config'] = (
                _wrap_fixed_range(
                    layer_config['config']['quantize_config'],
                    num_bits=num_bits,
                    init_min=output_min,
                    init_max=output_max,
                    narrow_range=narrow_range))
          break

    model = keras.Model.from_config(config)
  else:
    if (len(config['layers']) < 1 or
        config['layers'][1]['class_name'] != 'QuantizeLayer'):
      raise ValueError('`model` should be already quantized.')
    config['layers'][1]['config'][
        'quantizer'] = serialized_fixed_input_quantizer
    if 'quantize_config' in config['layers'][-1]['config']:
      config['layers'][-1]['config']['quantize_config'] = (
          _wrap_fixed_range(
              config['layers'][-1]['config']['quantize_config'],
              num_bits=num_bits,
              init_min=output_min,
              init_max=output_max,
              narrow_range=narrow_range))

    model = keras.Sequential.from_config(config)

  return model


def _is_functional_model(model):
  return (isinstance(model, keras.Model)
          and not isinstance(model, keras.Sequential)
          and model._is_graph_network)    # pylint: disable=protected-access


def remove_input_range(model):
  """Remove the input range.

  Example:

  ```python
  model = keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      quantize_annotate_layer(layers.Dense(2, activation='sigmoid'))
  ])
  with quantize.quantize_scope():
    model = quantize_annotate_model(model)
    model = quantize_apply(model)
    model = remove_input_range(model)
  ```

  In certain cases, a desired input range is not required if the model itself is
  internally used.

  Args:
    model: A `tf.keras` Sequential or Functional model which has been quantized.

  Returns:
    Returns a new `tf.keras` model removed input range.
  """
  config = model.get_config()
  no_input_quantizer = quantizers.NoQuantizer()
  serialized_input_quantizer = tf.keras.utils.serialize_keras_object(
      no_input_quantizer)

  if _is_functional_model(model):
    input_layer_list = _nested_to_flatten_node_data_list(config['input_layers'])
    for layer_config in config['layers']:
      input_name = _unwrap_first_input_name(layer_config['inbound_nodes'])
      if input_name is None:
        continue

      for input_layer in input_layer_list:
        if input_name == input_layer[0]:
          layer_config['config']['quantizer'] = serialized_input_quantizer
          break

    model = keras.Model.from_config(config)
  else:
    if (len(config['layers']) < 1 or
        config['layers'][1]['class_name'] != 'QuantizeLayer'):
      raise ValueError('`model` should be already quantized.')
    config['layers'][1]['config'][
        'quantizer'] = serialized_input_quantizer

    model = keras.Sequential.from_config(config)

  return model
