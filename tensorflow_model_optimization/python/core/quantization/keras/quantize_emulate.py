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
def quantize_annotate(
    to_quantize,
    num_bits,
    narrow_range=True,
    symmetric=True,
    **kwargs):  # pylint: disable=invalid-name
  """Specify a layer or model to be quantized.

  This function does not apply an quantization emulation operations. It merely
  wraps the keras layer (or each layer in the model) with `QuantizeAnnotate`
  to note which layers need to be quantized.

  Args:
    to_quantize: Keras layer or model to be quantized.
    num_bits: Number of bits for quantization
    narrow_range: Whether to use the narrow quantization range [1; 2^num_bits
      - 1] or wide range [0; 2^num_bits - 1].
    symmetric: If true, use symmetric quantization limits instead of training
      the minimum and maximum of each quantization range separately.
    **kwargs: Additional keyword arguments to be passed to the keras layer.

  Returns:
    Keras layer wrapped with `QuantizeAnnotate` if layer is passed. Else,
    a new keras model with each layer in the model wrapped with
    `QuantizeAnnotate`.
  """

  def _add_quant_wrapper(layer):
    if isinstance(layer, quant_annotate.QuantizeAnnotate):
      return layer

    return quant_annotate.QuantizeAnnotate(layer, **quant_params)

  quant_params = {
      'num_bits': num_bits,
      'narrow_range': narrow_range,
      'symmetric': symmetric
  }

  if isinstance(to_quantize, keras.Model):
    return keras.models.clone_model(
        to_quantize, input_tensors=None, clone_function=_add_quant_wrapper)
  elif isinstance(to_quantize, keras.layers.Layer):
    quant_params.update(**kwargs)
    return quant_annotate.QuantizeAnnotate(to_quantize, **quant_params)
