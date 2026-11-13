# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.keras.compat import keras
from tensorflow_model_optimization.python.core.sparsity.keras.tools import evaluate_pruning


def _build_model(with_lambda):
  layers = [keras.layers.Dense(4, input_shape=(3,))]
  if with_lambda:
    layers.append(keras.layers.Lambda(lambda x: x * 2.0))
  layers.append(keras.layers.Dense(2))
  return keras.Sequential(layers)


class EvaluatePruningLoadModelTest(tf.test.TestCase, parameterized.TestCase):

  def _save_model(self, model, filename):
    path = os.path.join(self.create_tempdir().full_path, filename)
    model.save(path)
    return path

  @parameterized.parameters('model.h5', 'model.keras')
  def test_load_model_rejects_lambda_by_default(self, filename):
    path = self._save_model(_build_model(with_lambda=True), filename)

    with self.assertRaisesRegex(ValueError, 'Lambda'):
      evaluate_pruning.load_model(path)

  @parameterized.parameters('model.h5', 'model.keras')
  def test_load_model_allows_lambda_when_unsafe_allowed(self, filename):
    model = _build_model(with_lambda=True)
    path = self._save_model(model, filename)

    loaded_model = evaluate_pruning.load_model(
        path, allow_unsafe_deserialization=True
    )

    x = np.ones((1, 3), dtype=np.float32)
    self.assertAllClose(model.predict(x), loaded_model.predict(x))

  @parameterized.parameters('model.h5', 'model.keras')
  def test_load_model_loads_safe_model(self, filename):
    model = _build_model(with_lambda=False)
    path = self._save_model(model, filename)

    loaded_model = evaluate_pruning.load_model(path)

    x = np.ones((1, 3), dtype=np.float32)
    self.assertAllClose(model.predict(x), loaded_model.predict(x))

  def test_load_model_restores_safe_mode_state(self):
    unsafe_path = self._save_model(_build_model(with_lambda=True), 'model.h5')

    evaluate_pruning.load_model(unsafe_path, allow_unsafe_deserialization=True)

    # Allowing unsafe deserialization once must not leak to later loads.
    with self.assertRaisesRegex(ValueError, 'Lambda'):
      evaluate_pruning.load_model(unsafe_path)

  def test_run_rejects_lambda_model(self):
    path = self._save_model(_build_model(with_lambda=True), 'model.h5')
    output_dir = self.create_tempdir().full_path

    with self.assertRaisesRegex(ValueError, 'Lambda'):
      evaluate_pruning.run(
          path, output_dir, target_sparsity=0.5, block_size=(1, 1)
      )

    self.assertEmpty(os.listdir(output_dir))


if __name__ == '__main__':
  tf.test.main()
