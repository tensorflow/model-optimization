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
"""Util functions for PQAT"""

import tensorflow as tf


@tf.custom_gradient
def apply_sparsity_mask_to_weights_gradient(weights, sparsity_mask):
    # Here zero-out the weights gradient with sparsity mask,
    # it ensure sparsity of weights are preserved with PQAT
    def grad(weights_gradient):
        return tf.multiply(weights_gradient, sparsity_mask), None
    return weights, grad
