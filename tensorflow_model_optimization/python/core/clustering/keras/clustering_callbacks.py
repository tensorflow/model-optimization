# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Clustering Callbacks."""

import tensorflow as tf

from tensorflow_model_optimization.python.core.keras import compat


class ClusteringSummaries(tf.keras.callbacks.TensorBoard):
  """Helper class to create tensorboard summaries for the clustering progress.

    This class is derived from tf.keras.callbacks.TensorBoard and just adds
    functionality to write histograms with batch-wise frequency.

    Arguments:
        log_dir: The path to the directory where the log files are saved
        cluster_update_freq: determines the frequency of updates of the
          clustering histograms. Same behaviour as parameter update_freq of the
          base class, i.e. it accepts `'batch'`, `'epoch'` or integer.
  """

  def __init__(self, log_dir='logs', cluster_update_freq='epoch', **kwargs):
    super(ClusteringSummaries, self).__init__(log_dir=log_dir, **kwargs)

    if not isinstance(log_dir, str) or not log_dir:
      raise ValueError(
          '`log_dir` must be a non-empty string. You passed `log_dir`='
          '{input}.'.format(input=log_dir))

    self.cluster_update_freq = (1 if cluster_update_freq == 'batch' else
                                cluster_update_freq)

    if compat.is_v1_apis():  # TF 1.X
      self.writer = tf.compat.v1.summary.FileWriter(log_dir)
    else:  # TF 2.X
      self.writer = tf.summary.create_file_writer(log_dir)

    self.continuous_batch = 0

  def on_train_batch_begin(self, batch, logs=None):
    super().on_train_batch_begin(batch, logs)
    # Count batches manually to get a continuous batch count spanning
    # epochs, because the function parameter 'batch' is reset to zero
    # every epoch.
    self.continuous_batch += 1

  def on_train_batch_end(self, batch, logs=None):
    assert self.continuous_batch >= batch, (
        'Continuous batch count must always be greater or equal than the'
        'batch count from the parameter in the current epoch.')

    super().on_train_batch_end(batch, logs)

    if self.cluster_update_freq == 'epoch':
      return
    elif self.continuous_batch % self.cluster_update_freq != 0:
      return  # skip this batch

    self._write_summary()

  def on_epoch_end(self, epoch, logs=None):
    super().on_epoch_end(epoch, logs)
    if self.cluster_update_freq == 'epoch':
      self._write_summary()

  def _write_summary(self):
    with self.writer.as_default():
      for layer in self.model.layers:
        if not hasattr(layer, 'layer') or not hasattr(
            layer.layer, 'get_clusterable_weights'):
          continue  # skip layer
        clusterable_weights = layer.layer.get_clusterable_weights()
        if len(clusterable_weights) < 1:
          continue  # skip layers without clusterable weights
        prefix = 'clustering/'
        # Log variables
        for var in layer.variables:
          success = tf.summary.histogram(
              prefix + var.name, var, step=self.continuous_batch)
          assert success
