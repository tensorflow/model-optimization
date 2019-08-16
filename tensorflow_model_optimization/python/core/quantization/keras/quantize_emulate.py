# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Entry point for quantize emulation during training of models."""

from tensorflow.python import keras

from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate as quant_annotate
from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.quantize_emulate_wrapper import QuantizeEmulateWrapper


def QuantizeEmulate(to_quantize,
                    num_bits,
                    narrow_range=True,
                    symmetric=True,
                    **kwargs):
  """Use this function to emulate quantization on NN layers during training.

  The function accepts a single layer or multiple layers and handles them
  appropriately.

  Arguments:
      to_quantize: A single keras layer, list of keras layers, or a
        `tf.keras.Sequential` model.
      num_bits: Number of bits for quantization
      narrow_range: Whether to use the narrow quantization range [1; 2^num_bits
        - 1] or wide range [0; 2^num_bits - 1].
      symmetric: If true, use symmetric quantization limits instead of training
        the minimum and maximum of each quantization range separately.
      **kwargs: Additional keyword arguments.

  Returns:
      Wrapped layer with quantization applied.
  """

  def _QuantizeList(layers, **params):
    """Apply QuantizeEmulate wrapper to a list of layers.

    Args:
      layers: List of keras layers to apply QuantizeEmulate.
      **params: QuantizationParams for the entire list.

    Returns:
      List of layers wrapped with QuantizeEmulate.
    """
    wrapped_layers = []

    for layer in layers:
      # Already quantized. Simply use and return. This supports usage such as
      # model = QuantizeEmulate([
      #           Dense(),
      #           QuantizeEmulate(Dense(), layer_params)
      #           Dense()
      #         ], model_params)
      if isinstance(layer, QuantizeEmulateWrapper):
        wrapped_layers.append(layer)
        continue

      wrapped_layers.append(QuantizeEmulate(layer, **params))

    return wrapped_layers

  params = {
      'num_bits': num_bits,
      'narrow_range': narrow_range,
      'symmetric': symmetric
  }
  params.update(kwargs)

  if isinstance(to_quantize, list):
    return _QuantizeList(to_quantize, **params)
  elif isinstance(to_quantize, keras.Sequential):
    return keras.models.Sequential(_QuantizeList(to_quantize.layers, **params))
  elif isinstance(to_quantize, keras.layers.Layer):
    return QuantizeEmulateWrapper(to_quantize, **params)


# TODO(pulkitb): Enable lint naming is fixed and made consistent.
def quantize_annotate(to_quantize, **kwargs):  # pylint: disable=invalid-name
  """Specify a layer or model to be quantized.

  This function does not apply an quantization emulation operations. It merely
  wraps the keras layer (or each layer in the model) with `QuantizeAnnotate`
  to note which layers need to be quantized.

  Args:
    to_quantize: Keras layer or model to be quantized.
    **kwargs: Additional keyword arguments to be passed to the keras layer.

  Returns:
    Keras layer wrapped with `QuantizeAnnotate` if layer is passed. Else,
    a new keras model with each layer in the model wrapped with
    `QuantizeAnnotate`.
  """

  def _add_quant_wrapper(layer):
    if isinstance(layer, quant_annotate.QuantizeAnnotate):
      return layer

    return quant_annotate.QuantizeAnnotate(layer)

  if isinstance(to_quantize, keras.Model):
    return keras.models.clone_model(
        to_quantize, input_tensors=None, clone_function=_add_quant_wrapper)
  elif isinstance(to_quantize, keras.layers.Layer):
    # TODO(pulkitb): Since annotation for model and layer have different
    # parameters, we should likely remove support for layers here.
    return quant_annotate.QuantizeAnnotate(to_quantize, **kwargs)


def quantize_apply(model):
  """Apply quantization operations to a keras model.

  This function takes a keras model which has been annotated with
  `quantize_annotate` and constructs a new keras model in which each of the
  annotated layers have been quantized. The quantization process introduces
  new quantization ops in the Tensorflow graph to appropriately emulate
  quantization loss.

  Note that to exactly emulate quantization loss, certain graph/model
  transformations may be applied. This is required since the actual quantized
  kernel implementations may apply similar transformations.

  Args:
    model: A keras Sequential or Functional model which has been annotated
    with `quantize_annotate`.

  Returns:
    Returns a new cloned keras model in which the annotated layers have been
    quantized. All the existing layers are cloned.
  """

  if not isinstance(model, keras.Model):
    raise ValueError('Only a keras `Model` instance can be used.')

  if not isinstance(model, keras.Sequential) \
      and not model._is_graph_network:  # pylint: disable=protected-access
    raise ValueError('model should be either a keras.Sequential or a '
                     'keras functional model.')

  # Have at least 1 layer annotated with QuantizeAnnotate
  if not any(isinstance(layer, quant_annotate.QuantizeAnnotate)
             for layer in model.layers):
    raise ValueError('model does not contain any layers which have been '
                     'annotated with `quantize_annotate`. There are no layers '
                     'to quantize.')

  if not model.built:
    raise ValueError('quantization cannot be applied to a model which has not'
                     'been built yet. Please call `model.build(input_shape)`'
                     'before quantizing your model.')

  def _clone_model_with_weights(model_to_clone):
    cloned_model = keras.models.clone_model(model_to_clone)
    cloned_model.set_weights(model_to_clone.get_weights())

    return cloned_model

  def _apply_quantization(quant_annotate_layer):
    return QuantizeEmulateWrapper(
        quant_annotate_layer.layer,
        **(quant_annotate_layer.get_quantize_params()))

  # Create a copy of the model with the same weights. We can then quantize this
  # model without modifying the weights of the original model.
  model_copy = _clone_model_with_weights(model)

  def _add_quant_emulate_wrapper(layer):  # pylint: disable=missing-docstring
    if not isinstance(layer, quant_annotate.QuantizeAnnotate):
      return layer

    # Use QuantizeEmulate wrapper on annotated layer which actually
    # quantization ops.
    return _apply_quantization(layer)

  return keras.models.clone_model(
      model_copy, input_tensors=None, clone_function=_add_quant_emulate_wrapper)
