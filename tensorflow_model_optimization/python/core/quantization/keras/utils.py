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
# pylint: disable=protected-access
"""Quantization specific utilities for generating, saving, testing, and evaluating models."""

import tensorflow as tf

from tensorflow.python.keras import backend as K


def convert_keras_to_tflite(model_path,
                            output_path,
                            custom_objects=None,
                            is_quantized=True):
  """Convert Keras model to TFLite."""
  if custom_objects is None:
    custom_objects = {}

  converter = tf.lite.TFLiteConverter.from_keras_model_file(
      model_path, custom_objects=custom_objects)

  if is_quantized:
    converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {
        input_arrays[0]: (0., 255.)
    }  # mean, std_dev

  tflite_model = converter.convert()
  with open(output_path, 'wb') as f:
    f.write(tflite_model)
