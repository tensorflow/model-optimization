class Bernouilli(tf.keras.initializers.Initializer):
  """
  Initialization distributio following a Bernouilli process..
  """

  def __init__(self, p):
    """
    p: probability parameter of success (i.e. 1).
    """
    self.p = p

  def get_config(self):
    return {'p': self.p}

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def __call__(self, shape, dytpe=tf.dtypes.float32):
    """Number of zeros = np.ceil(sparsity * size) in expectation."""
    probs = tf.zeros(shape=list(shape)) + self.p
    uniform = tf.random.uniform(shape)
    initial = tf.less(uniform, probs)

    return tf.cast(initial, dtype=dtype)

class PermuteOnes(tf.keras.initializers.Initializer):
  """
  Initialization of a deterministically sparse matrix.
  """
  def __init__(self, p=None):
    """
    p: probability parameter of success (i.e. 1).
    If p is None, will sample randomly from uniform distribution for sparsity.
    """
    self.p = p if p else tf.random.uniform(())

  def get_n_ones(self, shape, dtype=tf.dtypes.float32):
    sparsity = self.p if self.p else 0.0
    return tf.math.ceil(sparsity * tf.cast(tf.math.reduce_sum(shape)), dtype=dtype)

  def __call__(self, shape, dtype=tf.dtypes.float32, seed=None):
    flat_mask = tf.reshape(tf.ones(shape), (-1,))
    num_ones = self.get_n_ones(shape, dtype)
    _indices = tf.cast(tf.reshape(tf.linspace(0, num_ones - 1, int(num_ones)), (-1,)), tf.int32)
    indices = tf.reshape(_indices, (-1, 1))
    updates = tf.ones_like(_indices)
    flat_shape = flat_mask.shape
    unshuffled_mask = tf.scatter_nd(indices, udpates, flat_shape)
    shuffled_mask = tf.random.shuffle(unshuffled_mas, seed=seed)

    return tf.reshape(shuffled_mask, shape)


class ErdosRenyi(tf.keras.Initializers.Initializer):
  """Initialization based on the Erdos-Renyi distribution."""
  def __init__(self, sparsity):
    self.sparsity = sparsity

  def __call__(self, shape, dtype=tf.dtypes.float32, seed=None):
    return
  

class ErdosRenyiKernel(tf.keras.Initializers.Initializer):
  """Initialization based on the Erdos-Renyi distribution for CNNs."""
  def __init__(self, sparsity, kernel_dim):
      self.sparsity = sparsity
      self.kernel_dim_scale = kernel_dim

  def __call__(self, shape, dtype=tf.dtypes.float32, seed=None):
    return
