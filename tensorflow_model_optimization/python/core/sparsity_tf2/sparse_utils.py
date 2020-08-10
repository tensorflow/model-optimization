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
"""Convenience functions for sparse training."""

class Bernouilli(tf.keras.initializers.Initializer):
  """
  Initialization distributio following a Bernouilli process..
  """

  def __init__(self, p):
    """
    p: probability parameter of success (i.e. 1).
    """
    if not (p >= 0. and p <= 1.):
      raise ValueError('p parameter must be a valid probability, i.e. in [0, 1].')
    self.p = p

  def get_config(self):
    return {'p': self.p}

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def __call__(self, shape, dytpe=tf.dtypes.float32):
    """Number of zeros = np.ceil(probability * size) in expectation."""
    probs = tf.zeros(shape=list(shape)) + self.p
    uniform = tf.random.uniform(shape)
    initial = tf.less(uniform, probs)

    return tf.cast(initial, dtype=dtype)

class PermuteOnes(tf.keras.initializers.Initializer):
  """
  Initialization of a deterministically sparse matrix.
  This initializer takes in an input ratio and sets exactly
  that ratio of the mask entries as ones  leaving the rest as zeros.
  The ones are detministically, randomly permmuted across the tensor.
  """
  def __init__(self, ratio=None):
    """
    ratio: the exact number of 1s sampled.
    If ratio is None, will sample randomly from uniform distribution for sparsity.
    """
    if ratio is not None and not (ratio >= 0. and ratio <= 1.):
      raise ValueError('ratio parameter must be a valid percentage, i.e. in [0, 1].')
    self.ratio = ratio if ratio else tf.random.uniform(())

  def get_n_ones(self, shape, dtype=tf.dtypes.float32):
    sparsity = self.ratio if self.ratio else 0.0
    return tf.math.ceil(sparsity * tf.cast(tf.math.reduce_sum(shape)), dtype=dtype)

  def __call__(self, shape, dtype=tf.dtypes.float32, seed=None):
    flat_mask = tf.reshape(tf.ones(shape), (-1,))
    num_ones = self.get_n_ones(shape, dtype)
    _indices = tf.cast(tf.reshape(tf.linspace(0, num_ones - 1, int(num_ones)), (-1,)), tf.int32)
    indices = tf.reshape(_indices, (-1, 1))
    updates = tf.ones_like(_indices)
    flat_shape = flat_mask.shape
    unshuffled_mask = tf.scatter_nd(indices, updates, flat_shape)
    shuffled_mask = tf.random.shuffle(unshuffled_mask, seed=seed)

    return tf.reshape(shuffled_mask, shape)


class ErdosRenyi(tf.keras.Initializers.Initializer):
  """Sparsity initialization based on the Erdos-Renyi distribution.
  Ensures that the none of the non-custom layers have a total parameter
  count as the one with uniform sparsities, i.e. the non sparse
  layers satisfy the following equation:
    eps * (p_1 * N_1 + p_2 * N_2) = (1 - sparsity) ** (N_1 + N_2)
  """
  def __init__(self, sparsity, erk_power=1.0):
    """
    sparsity: the network target overall sparsity.
    erk_power: the power of the erk ratio
    """
    self.sparsity = sparsity
    self.erk_power = erk_power

  def __call__(self, shape, dtype=tf.dtypes.float32, seed=None):
    return
    # TODO
  

class ErdosRenyiKernel(tf.keras.Initializers.Initializer):
  """Initialization based on the Erdos-Renyi distribution for CNNs."""
  def __init__(self, sparsity, kernel_dim):
      self.sparsity = sparsity
      self.kernel_dim_scale = kernel_dim

  def __call__(self, shape, dtype=tf.dtypes.float32, seed=None):
    return
