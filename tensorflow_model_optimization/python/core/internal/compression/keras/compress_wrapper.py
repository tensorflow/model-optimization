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
"""Wrappers for tfmot compression."""
import tensorflow.compat.v2 as tf


class KerasLayerWrapper(tf.keras.layers.Wrapper):
  """Keras Layer wrapper."""

  def __init__(self, layer, attr_names, **kwargs):
    self.attr_names = attr_names
    self.original_add_weight = layer.add_weight
    layer.add_weight = self.sublayer_add_weight
    super(KerasLayerWrapper, self).__init__(layer, **kwargs)

  def sublayer_add_weight(self, *args, **kwargs):
    name = None

    if args:
      name = args[0]
      if len(args) > 1:
        shape = args[1]

    if 'name' in kwargs:
      name = kwargs['name']

    if 'shape' in kwargs:
      shape = kwargs['shape']

    if name in self.attr_names:
      return tf.zeros(shape)  # dummy for shape.

    return self.original_add_weight(*args, **kwargs)

  def build(self, input_shape=None):
    if not self.layer.built:
      self.layer.build(input_shape)
      self.layer.built = True

    self.built = True

  def call(self, inputs, *args, **kwargs):
    for attr_name in self.attr_names:
      if attr_name in kwargs:
        var = kwargs[attr_name]
        del kwargs[attr_name]
      elif args:
        var = args[0]
        args = args[1:]
      else:
        raise ValueError(
            'args not found. {} {} {} {}'.format(
                attr_name, inputs, args, kwargs))
      setattr(self.layer, attr_name, var)

    return self.layer(inputs, *args, **kwargs)

  def get_config(self):
    config = {'attr_names': self.attr_names}
    base_config = super(KerasLayerWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class ModuleWrapper(object):
  """Module wrapper."""

  def __init__(self, constructor, var_names, setter):
    self.constructor = constructor
    self.var_names = var_names
    self.setter = setter
    self.tensor_dict = {}

  # See tf.variable_creator_scope
  def _tensor_creator(self, next_creator, **kwargs):
    """Creator returning a Tensor with initial value (i.e. not Variable.)"""
    del next_creator  # Unused.
    ret = kwargs['initial_value']
    name = kwargs['name']
    self.tensor_dict[name] = kwargs
    return ret

  def build(self, input_shape=None):
    del input_shape  # Unused.
    with tf.variable_creator_scope(self._tensor_creator):
      self.module = self.constructor()
    self.built = True

  def call(self, method, inputs, *args, **kwargs):
    """Call the internal module with input variable value."""
    for var_name in self.var_names:
      if var_name in kwargs:
        value = kwargs[var_name]
        del kwargs[var_name]
      elif args:
        value = args[0]
        args = args[1:]
      else:
        raise ValueError(
            'args not found. {} {} {} {}'.format(
                var_name, inputs, args, kwargs))
      self.setter(self.module, var_name, value)

    if 'training' in kwargs:
      del kwargs['training']  # FIXME

    caller = getattr(self.module, method)
    return caller(inputs, *args, **kwargs)
