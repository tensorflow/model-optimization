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
"""Train a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
"""
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

from tensorflow_model_optimization.python.core.quantization.keras.quantize_emulate import QuantizeEmulate

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

if tf.keras.backend.image_data_format() == 'channels_first':
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
  input_shape = (1, img_rows, img_cols)
else:
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

l = tf.keras.layers
quant_params = {'num_bits': 8}

model = tf.keras.Sequential([
    QuantizeEmulate(
        l.Conv2D(32, 5, padding='same', activation='relu'),
        input_shape=input_shape,
        **quant_params),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    QuantizeEmulate(
        l.Conv2D(64, 5, padding='same', activation='relu'), **quant_params),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.Flatten(),
    QuantizeEmulate(l.Dense(1024, activation='relu'), **quant_params),
    l.Dropout(0.4),
    QuantizeEmulate(l.Dense(num_classes, activation='softmax'), **quant_params)
])

# Dump graph to /tmp for verification on tensorboard.
graph_def = tf.get_default_graph().as_graph_def()
with open('/tmp/mnist_model.pbtxt', 'w') as f:
  f.write(str(graph_def))

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adadelta(),
    metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Export to Keras.
keras_file = '/tmp/quantized_mnist.h5'
tf.keras.models.save_model(model, keras_file)
