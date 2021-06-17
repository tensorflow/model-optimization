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
"""Pruning Policy tests."""

import distutils.version as version

import tensorflow as tf

from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_policy
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

keras = tf.keras
layers = keras.layers


class CompatGlobalAveragePooling2D(layers.GlobalAveragePooling2D):
  """GlobalAveragePooling2D in tf <= 2.5.0 doesn't support keepdims."""

  def __init__(self, *args, keepdims=False, **kwargs):
    self._compat = False
    if version.LooseVersion(tf.__version__) > version.LooseVersion('2.5.0'):
      super(CompatGlobalAveragePooling2D, self).__init__(
          *args, keepdims=keepdims, **kwargs)
    else:
      super(CompatGlobalAveragePooling2D, self).__init__(*args, **kwargs)
      self._compat = True
      self.keepdims = keepdims

  def call(self, inputs):
    if not self._compat:
      return super(CompatGlobalAveragePooling2D, self).call(inputs)

    if self.data_format == 'channels_last':
      return keras.backend.mean(inputs, axis=[1, 2], keepdims=self.keepdims)
    else:
      return keras.backend.mean(inputs, axis=[2, 3], keepdims=self.keepdims)


class PruningPolicyTest(tf.test.TestCase):
  INVALID_TO_PRUNE_START_LAYER_ERROR = (
      'Could not find `Conv2D 3x3` layer with stride 2x2, `input filters == 3`'
      ' and `VALID` padding and preceding `ZeroPadding2D` with `padding == 1` '
      'in all input branches of the model'
  )

  INVALID_TO_PRUNE_STOP_LAYER_ERROR = (
      'Could not find a `GlobalAveragePooling2D` layer with `keepdims = True` '
      'in all output branches')

  INVALID_TO_PRUNE_MIDDLE_LAYER_ERROR = (
      'Layer {} is not supported for the '
      'PruneForLatencyOnXNNPack pruning policy')

  INVALID_TO_PRUNE_UNBUILT_ERROR = (
      'Unbuilt models are not supported currently.')

  def setUp(self):
    super(PruningPolicyTest, self).setUp()

    self.params = {
        'pruning_schedule': pruning_schedule.ConstantSparsity(0.5, 0),
        'block_size': (1, 1),
        'block_pooling_type': 'AVG'
    }

  @staticmethod
  def _count_pruned_layers(model):
    count = 0
    for layer in model.submodules:
      if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
        count += 1
    return count

  def testPruneUnsupportedModelForLatencyOnXNNPackPolicyNoStartLayer(self):
    i = keras.Input(shape=(8, 8, 3))
    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
    )(i)
    x = layers.Conv2D(filters=16, kernel_size=[1, 1])(x)
    o = CompatGlobalAveragePooling2D(keepdims=True)(x)
    model = keras.Model(inputs=[i], outputs=[o])
    with self.assertRaises(ValueError) as e:
      _ = prune.prune_low_magnitude(
          model,
          pruning_policy=pruning_policy.PruneForLatencyOnXNNPack(),
          **self.params,
      )
    self.assertEqual(str(e.exception), self.INVALID_TO_PRUNE_START_LAYER_ERROR)

  def testPruneUnsupportedModelForLatencyOnXNNPackPolicyNoStopLayer(self):
    i = keras.Input(shape=(8, 8, 3))
    x = layers.ZeroPadding2D(padding=1)(i)
    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='valid',
    )(x)
    x = layers.Conv2D(filters=16, kernel_size=[1, 1])(x)
    o = CompatGlobalAveragePooling2D()(x)
    model = keras.Model(inputs=[i], outputs=[o])
    with self.assertRaises(ValueError) as e:
      _ = prune.prune_low_magnitude(
          model,
          pruning_policy=pruning_policy.PruneForLatencyOnXNNPack(),
          **self.params,
      )
    self.assertEqual(str(e.exception), self.INVALID_TO_PRUNE_STOP_LAYER_ERROR)

  def testPruneUnsupportedModelForLatencyOnXNNPackPolicyMiddleLayer(self):
    i = keras.Input(shape=(8, 8, 3))
    x = layers.ZeroPadding2D(padding=1)(i)
    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='valid',
    )(x)
    x = layers.Conv2D(filters=16, kernel_size=[1, 1])(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    o = CompatGlobalAveragePooling2D(keepdims=True)(x)
    model = keras.Model(inputs=[i], outputs=[o])
    with self.assertRaises(ValueError) as e:
      _ = prune.prune_low_magnitude(
          model,
          pruning_policy=pruning_policy.PruneForLatencyOnXNNPack(),
          **self.params,
      )
    self.assertEqual(
        str(e.exception),
        self.INVALID_TO_PRUNE_MIDDLE_LAYER_ERROR.format(
            layers.MaxPooling2D.__name__))

  def testPruneSequentialModelForLatencyOnXNNPackPolicy(self):
    # No InputLayer
    model = keras.Sequential([
        layers.ZeroPadding2D(padding=1),
        layers.Conv2D(
            filters=4,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='valid',
        ),
        layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same'),
        layers.Conv2D(filters=8, kernel_size=[1, 1]),
        CompatGlobalAveragePooling2D(keepdims=True),
    ])
    with self.assertRaises(ValueError) as e:
      _ = prune.prune_low_magnitude(
          model,
          pruning_policy=pruning_policy.PruneForLatencyOnXNNPack(),
          **self.params,
      )
    self.assertEqual(str(e.exception), self.INVALID_TO_PRUNE_UNBUILT_ERROR)

    # With InputLayer
    model = keras.Sequential([
        layers.ZeroPadding2D(padding=1, input_shape=(8, 8, 3)),
        layers.Conv2D(
            filters=4,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='valid',
        ),
        layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same'),
        layers.Conv2D(filters=8, kernel_size=[1, 1]),
        CompatGlobalAveragePooling2D(keepdims=True),
    ])
    pruned_model = prune.prune_low_magnitude(
        model,
        pruning_policy=pruning_policy.PruneForLatencyOnXNNPack(),
        **self.params,
    )
    self.assertEqual(self._count_pruned_layers(pruned_model), 1)

  def testPruneModelRecursivelyForLatencyOnXNNPackPolicy(self):
    original_model = keras.Sequential([
        layers.ZeroPadding2D(padding=1, input_shape=(8, 8, 3)),
        layers.Conv2D(
            filters=4,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='valid',
        ),
        keras.Sequential([
            layers.Conv2D(filters=8, kernel_size=[1, 1]),
            layers.Conv2D(filters=16, kernel_size=[1, 1]),
        ]),
        CompatGlobalAveragePooling2D(keepdims=True),
    ])
    pruned_model = prune.prune_low_magnitude(
        original_model,
        pruning_policy=pruning_policy.PruneForLatencyOnXNNPack(),
        **self.params)
    self.assertEqual(self._count_pruned_layers(pruned_model), 2)

  def testPruneFunctionalModelWithLayerReusedForLatencyOnXNNPackPolicy(self):
    # The model reuses the Conv2D() layer. Make sure it's only pruned once.
    i = keras.Input(shape=(8, 8, 3))
    x = layers.ZeroPadding2D(padding=1)(i)
    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='valid',
    )(x)
    conv_layer = layers.Conv2D(filters=16, kernel_size=[1, 1])
    x = conv_layer(x)
    x = conv_layer(x)
    o = CompatGlobalAveragePooling2D(keepdims=True)(x)
    model = keras.Model(inputs=[i], outputs=[o])
    pruned_model = prune.prune_low_magnitude(
        model,
        pruning_policy=pruning_policy.PruneForLatencyOnXNNPack(),
        **self.params,
    )
    self.assertEqual(self._count_pruned_layers(pruned_model), 1)

  def testFunctionalModelNoPruningLayersForLatencyOnXNNPackPolicy(self):
    i = keras.Input(shape=(8, 8, 3))
    x = layers.ZeroPadding2D(padding=1)(i)
    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='valid',
    )(x)
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)
    o = CompatGlobalAveragePooling2D(keepdims=True)(x)
    model = keras.Model(inputs=[i], outputs=[o])

    pruned_model = prune.prune_low_magnitude(
        model,
        pruning_policy=pruning_policy.PruneForLatencyOnXNNPack(),
        **self.params,
    )
    self.assertEqual(self._count_pruned_layers(pruned_model), 0)

  def testFunctionalModelForLatencyOnXNNPackPolicy(self):
    i1 = keras.Input(shape=(16, 16, 3))
    x1 = layers.ZeroPadding2D(padding=1)(i1)
    x1 = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='valid',
    )(x1)
    x1_1 = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(x1)
    x1_2 = layers.Conv2D(filters=16, kernel_size=[1, 1])(x1)
    x1 = layers.Add()([x1_1, x1_2])

    i2 = keras.Input(shape=(16, 16, 3))
    x2 = layers.ZeroPadding2D(padding=1)(i2)
    x2 = layers.Conv2D(
        filters=8,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='valid',
    )(x2)
    x2 = layers.ZeroPadding2D(padding=1)(x2)
    x2 = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='valid',
    )(x2)
    x2_1 = CompatGlobalAveragePooling2D(keepdims=True)(x2)
    x2_1 = layers.Conv2D(filters=32, kernel_size=[1, 1])(x2_1)
    x2_1 = layers.Activation('sigmoid')(x2_1)
    x2_2 = layers.Conv2D(filters=32, kernel_size=[1, 1])(x2)
    x2_2 = layers.UpSampling2D(interpolation='bilinear')(x2_2)
    x2 = layers.Multiply()([x2_1, x2_2])

    x2 = layers.Conv2D(filters=16, kernel_size=[1, 1])(x2)
    x = layers.Add()([x1, x2])
    x = CompatGlobalAveragePooling2D(keepdims=True)(x)

    o1 = layers.Conv2D(filters=7, kernel_size=[1, 1])(x)
    o2 = layers.Conv2D(filters=3, kernel_size=[1, 1])(x)
    model = keras.Model(inputs=[i1, i2], outputs=[o1, o2])

    pruned_model = prune.prune_low_magnitude(
        model,
        pruning_policy=pruning_policy.PruneForLatencyOnXNNPack(),
        **self.params,
    )
    self.assertEqual(self._count_pruned_layers(pruned_model), 6)

  def testPruneFunctionalModelAfterCloneForLatencyOnXNNPackPolicy(self):
    i = keras.Input(shape=(8, 8, 3))
    x = layers.ZeroPadding2D(padding=1)(i)
    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='valid',
    )(
        x)
    x = layers.Conv2D(filters=16, kernel_size=[1, 1])(x)
    o = CompatGlobalAveragePooling2D(keepdims=True)(x)
    original_model = keras.Model(inputs=[i], outputs=[o])

    cloned_model = tf.keras.models.clone_model(
        original_model, clone_function=lambda l: l)
    pruned_model = prune.prune_low_magnitude(
        cloned_model,
        pruning_policy=pruning_policy.PruneForLatencyOnXNNPack(),
        **self.params,
    )
    self.assertEqual(self._count_pruned_layers(pruned_model), 1)

  def testFunctionalModelWithTFOpsForLatencyOnXNNPackPolicy(self):
    i = keras.Input(shape=(8, 8, 3))
    x = layers.ZeroPadding2D(padding=1)(i)
    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='valid',
    )(x)
    residual = layers.Conv2D(filters=16, kernel_size=[1, 1])(x)
    x = x + residual
    x = x - residual
    x = x * residual
    x = tf.identity(x)
    o = CompatGlobalAveragePooling2D(keepdims=True)(x)
    model = keras.Model(inputs=[i], outputs=[o])

    pruned_model = prune.prune_low_magnitude(
        model,
        pruning_policy=pruning_policy.PruneForLatencyOnXNNPack(),
        **self.params,
    )
    self.assertEqual(self._count_pruned_layers(pruned_model), 1)


if __name__ == '__main__':
  tf.test.main()
