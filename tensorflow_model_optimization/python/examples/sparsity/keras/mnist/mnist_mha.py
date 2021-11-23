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
# pylint: disable=missing-docstring
"""Train a simple model with MultiHeadAttention layer on MNIST dataset
and prune it.
"""
import tensorflow as tf

from tensorflow_model_optimization.python.core.keras import test_utils as keras_test_utils
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_utils
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

tf.random.set_seed(42)

ConstantSparsity = pruning_schedule.ConstantSparsity

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# define model
input = tf.keras.layers.Input(shape=(28, 28))
x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=16, name='mha')(
    query=input, value=input
)
x = tf.keras.layers.Flatten()(x)
out = tf.keras.layers.Dense(10)(x)
model = tf.keras.Model(inputs=input, outputs=out)

# Train the digit classification model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

model.fit(
    train_images, train_labels, epochs=10, validation_split=0.1,
)

score = model.evaluate(test_images, test_labels, verbose=0)
print('Model test loss:', score[0])
print('Model test accuracy:', score[1])

# Define parameters for pruning

batch_size = 128
epochs = 3
validation_split = 0.1  # 10% of training set will be used for validation set.

callbacks = [
    pruning_callbacks.UpdatePruningStep(),
    pruning_callbacks.PruningSummaries(log_dir='/tmp/logs')
]

pruning_params = {
      'pruning_schedule': ConstantSparsity(0.75, begin_step=2000, frequency=100)
}

model_for_pruning = prune.prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

model_for_pruning.fit(
    train_images,
    train_labels,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=validation_split,
)

score = model_for_pruning.evaluate(test_images, test_labels, verbose=0)
print('Pruned model test loss:', score[0])
print('Pruned model test accuracy:', score[1])
