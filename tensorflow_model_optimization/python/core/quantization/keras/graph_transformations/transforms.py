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
"""Defines core classes for expressing keras model transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


class LayerPattern(object):
  """Defines a tree sub-graph pattern of Keras layers to match in a model.

  `LayerPattern` can be used to describe various common patterns in model
  graphs that we need to find.

  Examples:
    Matches a Conv+BN+ReLU6 pattern.
    pattern = LayerPattern('ReLU', {'max_value': 6.0}, [
        LayerPattern('BatchNormalization', {}, [
            LayerPattern('Conv2D', {} [])
        ])
    ])

    Matches multiple Conv2Ds feeding into a Concat.
    pattern = LayerPattern('Concat', {}, [
        LayerPattern('Conv2D', {}, []),
        LayerPattern('Conv2D', {}, [])
    )
  """

  def __init__(self, class_name, config=None, inputs=None):
    """Construct pattern to match.

    Args:
      class_name: Type of keras layer (such as Conv2D, Dense etc.)
      config: Map of arguments of the layer to match. For eg., for ReLU(6.0)
          it would be {'max_value': 6.0}.
      inputs: input layers to the layer.
    """
    if config is None:
      config = {}
    if inputs is None:
      inputs = []

    self.class_name = class_name
    self.config = config
    self.inputs = inputs


class LayerNode(object):
  """Represents a Node in a tree containing a layer.

  `LayerNode` is used to represent a tree of layers in a model. It contains
  config which describes the layer, and other input layers feeding into it.

  It is used as a generic class to represent both sets of layers which have
  been found in a model, and layers which should be replaced inside the model.
  """

  def __init__(self, layer, input_layers):
    """Construct a LayerNode representing a tree of layers.

    Args:
      layer: layer config of this node.
      input_layers: List of `LayerNode`s that feed into this layer.
    """
    self.layer = layer
    self.input_layers = input_layers

  def __str__(self):
    return '{} <- [{}]'.format(
        self.layer,
        ', '.join([str(input_layer) for input_layer in self.input_layers]))


@six.add_metaclass(abc.ABCMeta)
class Transform(object):
  """Defines a transform to be applied to a keras model graph.

  A transform is a combination of 'Find + Replace' which describes how to find
  a pattern of layers in a model, and what to replace those layers with.

  A pattern is described using `LayerPattern`. The replacement function receives
  a `LayerNode` which contains the matched layers and should return a
  `LayerNode` which contains the set of layers which replaced the matched
  layers.
  """

  @abc.abstractmethod
  def pattern(self):
    """Return the `LayerPattern` to find in the model graph."""
    raise NotImplementedError()

  @abc.abstractmethod
  def replacement(self, match_layer):
    """Generate a replacement sub-graph for the matched sub-graph.

    The fundamental constraint of the replacement is that the replacement
    sub-graph should consume the same input tensors as the original sub-graph
    and also produce a final list of tensors which are same in number and shape
    as the original sub-graph. Not following this could crash model creation,
    or introduce bugs in the new model graph.

    TODO(pulkitb): Consider adding list of input layers feeding into the
    sub-graph, and output layers feeding from the tip of the tree as parameters.
    These would be needed for complex replace cases.

    Args:
      match_layer: Matched sub-graph based on `self.pattern()`.
    """
    raise NotImplementedError()

  def custom_objects(self):
    """Dictionary of custom objects introduced by the `replacement` function.

    A `Transform` may introduce custom Classes and types unknown to Keras. This
    function should return a dictionary containing these objects in case such
    types are introduced. It allows model construction to serialize/deserialize
    these objects.

    Returns:
      Custom objects introduced by the transform as a dictionary.
    """
    return {}
