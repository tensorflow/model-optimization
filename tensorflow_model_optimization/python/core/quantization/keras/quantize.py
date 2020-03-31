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

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate as quantize_annotate_mod
from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config as quantize_config_mod
from tensorflow_model_optimization.python.core.quantization.keras import quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras import quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.layers import conv_batchnorm

keras = tf.keras


def quantize_scope(*args):
  """Required scope to deserialize quantized models stored in tf.keras h5 format.

  Args:
    *args: Variable length list of dictionaries of name, class pairs to add to
      the scope created by this method.

  Returns:
    Object of type `CustomObjectScope` with quantization objects included.

  Example:

  ```python
  tf.keras.models.save_model(quantized_model, keras_file)

  with quantize_scope():
    loaded_model = tf.keras.models.load_model(keras_file)
  ```
  """
  quantization_objects = {
      'QuantizeAnnotate': quantize_annotate_mod.QuantizeAnnotate,
      'QuantizeAwareActivation':
          quantize_aware_activation.QuantizeAwareActivation,
      'NoOpActivation': quantize_aware_activation.NoOpActivation,
      'QuantizeWrapper': quantize_wrapper.QuantizeWrapper,
      'QuantizeLayer': quantize_layer.QuantizeLayer,
      # TODO(tf-mot): add way for different quantization schemes to modify this.
      '_DepthwiseConvBatchNorm2D': conv_batchnorm._DepthwiseConvBatchNorm2D,  # pylint: disable=protected-access
      '_ConvBatchNorm2D': conv_batchnorm._ConvBatchNorm2D  # pylint: disable=protected-access
  }
  quantization_objects.update(default_8bit_quantize_registry._types_dict())  # pylint: disable=protected-access
  quantization_objects.update(quantizers._types_dict())  # pylint: disable=protected-access

  return tf.keras.utils.custom_object_scope(*(args + (quantization_objects,)))


# TODO(tfmot): link to docs to explain what quantization implementation means.
def quantize_model(to_quantize):
  """Quantize a whole tf.keras model with the default quantization implementation.

  To be more precise, `quantize_model` creates a model that emulates
  quantization during training and stores information that downstream
  tools will use to produce actually quantized models.

  Quantize a model:

  ```python
  model = quantize_model(
      keras.Sequential([
          layers.Dense(10, activation='relu', input_shape=(100,)),
          layers.Dense(2, activation='sigmoid')
      ]))
  ```

  Note that this function removes the optimizer from the original model.
  Additionally, training the model returned by `quantize_model` will not affect
  the weights of the original model.

  Args:
    to_quantize: tf.keras model to be quantized. It can have pre-trained
      weights.

  Returns:
    Returns a new tf.keras model prepared for quantization.
  """
  if to_quantize is None:
    raise ValueError('`to_quantize` cannot be None')

  if not isinstance(to_quantize, keras.Model):
    raise ValueError(
        '`to_quantize` can only be a `tf.keras.Model` instance. Use '
        'the `quantize_annotate_layer` API to handle individual layers.'
        'You passed an instance of type: {input}.'.format(
            input=to_quantize.__class__.__name__))

  if not isinstance(to_quantize, keras.Sequential) \
      and not to_quantize._is_graph_network:  # pylint: disable=protected-access
    raise ValueError(
        '`to_quantize` can only either be a tf.keras Sequential or '
        'Functional model.')

  annotated_model = quantize_annotate_model(to_quantize)
  return quantize_apply(annotated_model)


def quantize_annotate_model(to_annotate):
  """Annotate a model to be quantized.

  This function does not actually quantize anything. It is merely to specify
  that the model needs to be quantized.

  This function is intended to be used in conjunction with the
  `quantize_annotate_layer`
  API. It's otherwise simpler to use `quantize_model`.

  Annotate a model while overriding the default behavior for one layer:

  ```python
  quantize_config = MyDenseQuantizeConfig()

  model = quantize_annotate_model(keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      quantize_annotate_layer(layers.Dense(2, activation='sigmoid'),
      quantize_config=quantize_config)
  ])))
  ```

  Note that this function removes the optimizer from the original model.

  Args:
    to_annotate: tf.keras model to annotate to be quantized.

  Returns:
    New tf.keras model with each layer in the model wrapped with
    `QuantizeAnnotate`.
  """
  if to_annotate is None:
    raise ValueError('`to_annotate` cannot be None')

  if not isinstance(to_annotate, keras.Model):
    raise ValueError(
        '`to_annotate` can only be a `tf.keras.Model` instance. Use '
        'the `quantize_annotate_layer` API to handle individual layers. '
        'You passed an instance of type: {input}.'.format(
            input=to_annotate.__class__.__name__))

  if not isinstance(to_annotate, keras.Sequential) \
      and not to_annotate._is_graph_network:  # pylint: disable=protected-access
    raise ValueError(
        '`to_annotate` can only either be a tf.keras Sequential or '
        'Functional model.')

  def _add_quant_wrapper(layer):
    """Add annotation wrapper."""
    # Already annotated layer. No need to wrap.
    if isinstance(layer, quantize_annotate_mod.QuantizeAnnotate):
      return layer

    if isinstance(layer, tf.keras.Model):
      raise ValueError(
          'Quantizing a tf.keras Model inside another tf.keras Model is not supported.'
      )

    return quantize_annotate_mod.QuantizeAnnotate(layer)

  return keras.models.clone_model(
      to_annotate, input_tensors=None, clone_function=_add_quant_wrapper)


def quantize_annotate_layer(to_annotate, quantize_config=None):
  """Annotate a layer to be quantized.

  This function does not actually quantize anything. It is merely to specify
  that the layer needs to be quantized.

  Annotate a layer:

  ```python
  model = keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      quantize_annotate_layer(layers.Dense(2, activation='sigmoid'))
  ]))
  ```

  Note that this function removes the optimizer from the original model.

  Args:
    to_annotate: tf.keras layer to annotate to be quantized.
    quantize_config: `QuantizeConfig` to quantize layer.

  Returns:
    tf.keras layer wrapped with `QuantizeAnnotate`.
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


def quantize_apply(model):
  """Introduce quantization operations to a tf.keras model.

  This function takes a tf.keras model which has been annotated with
  `quantize_annotate` and constructs a new model in which each of the
  annotated layers will ultimately be quantized. The new quantization
  operations enable the model to **emulate* quantization during training
  and store information that downstream tools will use to produce
  an actually quantized model.

  Apply quantization to a model:

  ```python
  model = quantize_apply(annotated_model)
  ```

  Note that this function removes the optimizer from the original model.
  Additionally, training the model returned by `quantize_apply` will not affect
  the weights of the original model.

  Args:
    model: A tf.keras Sequential or Functional model which has been annotated
    with `quantize_annotate`. It can have pre-trained weights.

  Returns:
    Returns a new tf.keras model in which the annotated layers have been
    prepared for quantization.
  """
  if model is None:
    raise ValueError('`model` cannot be None')

  if not isinstance(model, keras.Model):
    raise ValueError('`model` can only be a `tf.keras.Model` instance.'
                     'You passed an instance of type: {input}.'.format(
                         input=model.__class__.__name__))

  if not isinstance(model, keras.Sequential) \
      and not model._is_graph_network:  # pylint: disable=protected-access
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

    def _unwrap(layer):
      if not isinstance(layer, quantize_annotate_mod.QuantizeAnnotate):
        return layer

      annotate_wrapper = layer
      layer_quantize_map[annotate_wrapper.layer.name] = {
          'quantize_config': annotate_wrapper.quantize_config
      }
      return annotate_wrapper.layer

    unwrapped_model = keras.models.clone_model(
        model_to_unwrap, input_tensors=None, clone_function=_unwrap)

    return unwrapped_model, layer_quantize_map

  def _quantize(layer):  # pylint: disable=missing-docstring
    if layer.name not in layer_quantize_map:
      return layer

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
    return quantize_wrapper.QuantizeWrapper(layer, quantize_config)

  # 1. Create a copy of the model with the same weights. This ensures
  # modifications don't affect the original model, or its weights.
  model_copy = _clone_model_with_weights(model)

  # 2. Remove QuantizeAnnotate wrappers from the layers in the model. This
  # extracts the original model structure (easier to transform), and
  # stores relevant quantization information in a map.
  unwrapped_model, layer_quantize_map = _extract_original_model(model_copy)
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
  quantize_transform = \
    default_8bit_quantize_layout_transform.QuantizeLayoutTransform()
  # layer_quantize_map gets modified by the transformations.
  transformed_model, layer_quantize_map = quantize_transform.apply(
      unwrapped_model, layer_quantize_map)

  # TODO(pulkitb): Think more about how to introduce Default specific code.
  quantize_registry = default_8bit_quantize_registry.QuantizeRegistry(
  )

  # 4. Actually quantize all the relevant layers in the model. This is done by
  # wrapping the layers with QuantizeWrapper, and passing the associated
  # `QuantizeConfig`.

  return keras.models.clone_model(
      transformed_model, input_tensors=None, clone_function=_quantize)
