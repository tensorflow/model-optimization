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
"""Tool to display info how model has been pruned.

This tool is used to display sparsity for each layer of the model and
type of sparsity that has been used to prune them: unstructured or
sparsity 2x4.
"""

from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf
import numpy as np

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_utils

_FILE_PATH = flags.DEFINE_string('model_tflite', "/tmp/mnist_2x4.tflite", 'TFLite model file')

# Dont check layer if its name has one of word from this list.
IGNORE_LIST = [
    "relu",
    "pooling",
    "reshape",
    "identity",
    "input",
    "add",
    "flatten"
]


def ignore_tensor(details, ignore_list):
  """Returns boolean that indicates whether to ignore the tensor."""
  name = details["name"].casefold()
  if not name:
    return True
  for to_ignore in ignore_list:
    if to_ignore in name:
      return True
  return False


def calculate_sparsity(weights):
  """Returns sparsity of the given weights tensor."""
  number_of_weights = np.size(weights)
  number_of_non_zero_weights = np.count_nonzero(weights)
  sparsity = 1.0 - float(number_of_non_zero_weights)/number_of_weights\
    if number_of_non_zero_weights != 0 else 1.0
  return sparsity


def print_info(name, shape, sparsity, type, applicable):
  """Prints information for the layer."""
  print("{}: shape: {}, sparsity: {}, 2x4: {}, applicable: {}".\
    format(name, shape, sparsity, type, applicable))


def run(input_tflite_path):
  """Checks type of sparsity for each layer of the model."""

  interpreter = tf.lite.Interpreter(model_path=input_tflite_path)
  interpreter.allocate_tensors()

  details = interpreter.get_tensor_details()
  # Don't consider layers that can't be pruned.
  details = [x for x in details if not ignore_tensor(x, IGNORE_LIST)]

  for detail in details:
    name = detail["name"]
    shape = detail["shape"]
    weights = interpreter.tensor(detail["index"])()

    is_applicable_2x4 = pruning_utils.check_if_applicable_sparsity_2x4(weights)
    is_pruned_2x4 = pruning_utils.is_pruned_2x4(weights) if is_applicable_2x4\
      else False
    sparsity = calculate_sparsity(weights)

    print_info(name, shape, sparsity, is_pruned_2x4, is_applicable_2x4)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  run(_FILE_PATH.value)


if __name__ == '__main__':
  app.run(main)
