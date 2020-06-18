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
# pylint: disable=missing-docstring
"""Injects sparsity to any model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
# import g3
import numpy as np
import tensorflow as tf

# b/(139939526): update to use public API.
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils

from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer
from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_impl
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

def inject_sparsity(model, pruning_config_from_layer):
  """
  Returns a new functional model (TODO(xwinxu): currently only supports functional/sequantial models)
  with a built pruning wrapper.

  Args:
    model: keras Functional model
    pruning_config_from_layer: Callable that takes a layer and returns the relevant hparam configs (user defined)
    TODO(xwinxu): PruningConfig class for pruning methods beyond just magnitude based
  """
  def clone_fn(layer):
    config = pruning_config_from_layer(layer)
    if config:
      layer = pruning_wrapper.PruneLowMagnitude(layer, **config) # TODO: rename PruneLowMagnitude
    
    return layer

  return tf.keras.models.clone_model(model, clone_function=clone_fn)
