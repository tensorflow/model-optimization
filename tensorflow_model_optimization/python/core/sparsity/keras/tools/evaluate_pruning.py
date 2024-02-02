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
"""Tool to quickly prune a Keras model for evaluation purpose.

Prunes the model with the given spasity parameters, without retraining. Will
output a converted TFLite model for both pruned and unpruned versions.

This tool is intented to produce sparsified models for evaluating the
performance benefits (model size, inference time, …) of pruning. Since the
sparsity is applied in one shot, without retraining, the accuracy of the
resulting model will be severly degraded.
"""

from __future__ import print_function

import os
import tempfile
import textwrap
import zipfile

from absl import app
from absl import flags
import tensorflow as tf

from tensorflow_model_optimization.python.core.keras.compat import keras
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras.tools import sparsity_tooling


_MODEL_PATH = flags.DEFINE_string('model', None, 'Keras model file to prune')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'Output directory')
_SPARSITY = flags.DEFINE_float(
    'sparsity',
    0.8,
    'Target sparsity level, as float in [0,1] interval',
    lower_bound=0,
    upper_bound=1)
_BLOCK_SIZE = flags.DEFINE_string(
    'block_size', '1,1',
    'Comma-separated dimensions (height,weight) of the block sparsity pattern.'
)


def _parse_block_size_flag(value):
  height_str, weight_str = value.split(',')
  return int(height_str), int(weight_str)


@flags.validator(_BLOCK_SIZE.name)
def _check_block_size(flag_value):
  try:
    _parse_block_size_flag(flag_value)
    return True
  except:
    raise flags.ValidationError('Invalid block size value "%s".' % flag_value)


def convert_to_tflite(keras_model, output_path):
  """Converts the given Keras model to TFLite and write it to a file."""
  converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
  converter.optimizations = {tf.lite.Optimize.EXPERIMENTAL_SPARSITY}

  with open(output_path, 'wb') as out:
    out.write(converter.convert())


def get_gzipped_size(model_path):
  """Measures the compressed size of a model."""
  with tempfile.TemporaryFile(suffix='.zip') as zipped_file:
    with zipfile.ZipFile(
        zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
      f.write(model_path)

    zipped_file.seek(0, 2)
    return os.fstat(zipped_file.fileno()).st_size


def pruned_model_filename(sparsity, block_size):
  """Produces a human-readable name including sparsity parameters."""
  return 'pruned_model_sparsity_%.2f_block_%s.tflite' % (
      sparsity, '%dx%d' % block_size)


def run(input_model_path, output_dir, target_sparsity, block_size):
  """Prunes the model and converts both pruned and unpruned versions to TFLite."""

  print(textwrap.dedent("""\
    Warning: The sparse models produced by this tool have poor accuracy. They
             are not intended to be served in production, but to be used for
             performance benchmarking."""))

  input_model = keras.models.load_model(input_model_path)

  os.makedirs(output_dir, exist_ok=True)
  unpruned_tflite_path = os.path.join(
      output_dir, 'unpruned_model.tflite')
  pruned_tflite_path = os.path.join(
      output_dir, pruned_model_filename(target_sparsity, block_size))

  # Convert to TFLite without pruning
  convert_to_tflite(input_model, unpruned_tflite_path)

  # Prune and convert to TFLite
  pruned_model = sparsity_tooling.prune_for_benchmark(
      keras_model=input_model,
      target_sparsity=target_sparsity,
      block_size=block_size)
  stripped_model = prune.strip_pruning(pruned_model)  # Remove pruning wrapper
  convert_to_tflite(stripped_model, pruned_tflite_path)

  # Measure the compressed size of unpruned vs pruned TFLite models
  unpruned_compressed_size = get_gzipped_size(unpruned_tflite_path)
  pruned_compressed_size = get_gzipped_size(pruned_tflite_path)
  print('Size of gzipped TFLite models:')
  print(' * Unpruned : %.2fMiB' % (unpruned_compressed_size / (2.**20)))
  print(' * Pruned   : %.2fMiB' % (pruned_compressed_size / (2.**20)))
  print('       diff : %d%%' %
        (100. * (pruned_compressed_size - unpruned_compressed_size) /
         unpruned_compressed_size))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  block_size = _parse_block_size_flag(_BLOCK_SIZE.value)
  run(_MODEL_PATH.value, _OUTPUT_DIR.value, _SPARSITY.value, block_size)


if __name__ == '__main__':
  flags.mark_flag_as_required(_MODEL_PATH.name)
  flags.mark_flag_as_required(_OUTPUT_DIR.name)

  app.run(main)
