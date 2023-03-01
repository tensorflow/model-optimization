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
"""Util functions for weight clustering."""
import tensorflow as tf


def _type_model(model):
  """Check type of the model: Sequential/Functional, Layer or Subclassed.

  Args:
    model : provided model to check
  Returns:
    [tuple]: (is_sequential_or_functional,
      is_keras_layer, is_subclassed_model)
  """
  # pylint:disable=protected-access
  is_sequential_or_functional = isinstance(
      model, tf.keras.Model) and (isinstance(model, tf.keras.Sequential) or
                                  model._is_graph_network)

  is_keras_layer = isinstance(
      model, tf.keras.layers.Layer) and not isinstance(model, tf.keras.Model)

  is_subclassed_model = isinstance(model, tf.keras.Model) and (
      not model._is_graph_network)

  return (is_sequential_or_functional, is_keras_layer, is_subclassed_model)


def strip_clustering_cqat(to_strip):
  """Strip clustering variables from the model.

  During cluster-preserve quantization aware training (CQAT), centroids,
  cluster associations, and original weights are added to the training graph.
  After the CQAT is done, these variables should be removed and the layer
  with the clustered weights should be restored.

  Arguments:
      to_strip: A `tf.keras.Model` instance with clustered layers or a
      `tf.keras.layers.Layer` instance

  Returns:
    A keras model or layer with clustering variables removed.

  Raises:
    ValueError: if the model is not a `tf.keras.Model` instance.
    NotImplementedError: if the model is a subclassed model.

  """
  if not isinstance(to_strip, tf.keras.Model) and not isinstance(
      to_strip, tf.keras.layers.Layer):
    raise ValueError(
        ('Expected to_strip to be a `tf.keras.Model` or'
         '`tf.keras.layers.Layer` instance but got: '), to_strip)

  def _strip_clustering_ops(layer):
    if isinstance(layer, tf.keras.Model):
      return tf.keras.models.clone_model(
          layer,
          input_tensors=None,
          clone_function=_strip_clustering_ops)

    # set the attributes of the layer to the result after cqat
    # and remove all other variables, we do not remove the
    # quantization aware training wrapper in this step
    # so that we can utilize the ranges in tflite converter

    # we only handle conv2d and dense layers here
    if hasattr(layer, 'layer'):
      # pylint:disable=protected-access
      if 'depthwise' not in layer.layer.name:
        if isinstance(layer.layer,
                      (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
          new_variables = []
          for v in layer._trainable_weights:
            if 'cluster_centroids_tf' in v.name or (
                'ori_weights_vars_tf' in v.name):
              continue
            new_variables.append(v)
          layer._trainable_weights = new_variables

          new_variables = []
          for v in layer._non_trainable_weights:
            if 'pulling_indices_tf' in v.name:
              continue
            new_variables.append(v)
          layer._non_trainable_weights = new_variables

    return layer

  (is_sequential_or_functional, is_keras_layer,
   is_subclassed_model) = _type_model(to_strip)

  # Just copy the model with the right callback
  if is_sequential_or_functional:
    return tf.keras.models.clone_model(
        to_strip, input_tensors=None, clone_function=_strip_clustering_ops)
  elif is_keras_layer:
    if isinstance(to_strip, tf.keras.layers.Layer):
      return _strip_clustering_ops(to_strip)
  elif is_subclassed_model:
    to_strip_model = to_strip.model
    # pylint:disable=protected-access
    for i, layer in enumerate(to_strip_model._self_tracked_trackables):
      to_strip_model._self_tracked_trackables[i] = _strip_clustering_ops(
          layer=layer)
    return to_strip_model
  else:
    raise ValueError(
        ' Strip clustering cannot be applied. You passed '
        'an object of type: {input}.'.
        format(input=to_strip.__class__.__name__))
