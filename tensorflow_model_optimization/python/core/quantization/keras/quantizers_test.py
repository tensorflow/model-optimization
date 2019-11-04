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
"""Tests for Quantizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

from tensorflow_model_optimization.python.core.quantization.keras import quantizers


@parameterized.parameters(
    quantizers.LastValueQuantizer, quantizers.MovingAverageQuantizer)
class QuantizersTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(QuantizersTest, self).setUp()
    self.quant_params = {
        'num_bits': 8,
        'per_axis': False,
        'symmetric': False,
        'narrow_range': False
    }

  @staticmethod
  def _test_quantizer(quantizer):
    with session.Session(graph=ops.Graph()) as sess:
      inputs = variable_scope.get_variable(
          'inputs', dtype=dtypes.float32,
          initializer=np.array([[-1.0, 0.5], [0.0, 1.0]], dtype=np.float32))
      min_var = variables.Variable(0.0)
      max_var = variables.Variable(0.0)

      kwargs = {'min_var': min_var, 'max_var': max_var}
      quant_tensor = quantizer(inputs, step=0, training=True, **kwargs)

      sess.run(variables.global_variables_initializer())
      results = sess.run(quant_tensor)
      min_max_values = sess.run([min_var, max_var])

      # TODO(pulkitb): Assert on expected values for testing.
      # Since the underlying code is already tested in quant_ops_test.py, this
      # just ensures the Quantizers code is wired properly.
      print('Result: ', results)
      print('min_var: ', min_max_values[0])
      print('max_var: ', min_max_values[1])

  def testQuantizer(self, quantizer_type):
    quantizer = quantizer_type(**self.quant_params)

    self._test_quantizer(quantizer)

  def testSerialization(self, quantizer_type):
    quantizer = quantizer_type(**self.quant_params)

    expected_config = {
        'class_name': quantizer_type.__name__,
        'config': {
            'num_bits': 8,
            'per_axis': False,
            'symmetric': False,
            'narrow_range': False
        }
    }
    serialized_quantizer = serialize_keras_object(quantizer)

    self.assertEqual(expected_config, serialized_quantizer)

    quantizer_from_config = deserialize_keras_object(
        serialized_quantizer,
        module_objects=globals(),
        custom_objects=quantizers._types_dict())

    self.assertEqual(quantizer, quantizer_from_config)


if __name__ == '__main__':
  test.main()
