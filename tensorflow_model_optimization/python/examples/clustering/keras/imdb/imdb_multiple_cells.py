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
"""End-to-end tests for StackedRNNCells.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF+LogReg.
"""

from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.keras.compat import keras
from tensorflow_model_optimization.python.examples.clustering.keras.imdb.imdb_utils import cluster_train_eval_strip
from tensorflow_model_optimization.python.examples.clustering.keras.imdb.imdb_utils import prepare_dataset


max_features = 20000
maxlen = 100  # cut texts after this number of words
batch_size = 32

x_train, y_train, x_test, y_test = prepare_dataset()

print("Build a model with the StackedRNNCells with LSTMCell...")
model = keras.models.Sequential()

model.add(keras.layers.Embedding(max_features, 128, input_length=maxlen))
model.add(
    keras.layers.RNN(
        keras.layers.StackedRNNCells(
            [keras.layers.LSTMCell(128) for _ in range(2)]
        )
    )
)
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation("sigmoid"))

test_case = "StackedRNNCells_LSTMCell"
cluster_train_eval_strip(
    model, x_train, y_train, x_test, y_test, batch_size, test_case)

print("Build a model with the Bidirectional wrapper with LSTM layer...")
model = keras.models.Sequential()

model.add(keras.layers.Embedding(max_features, 128, input_length=maxlen))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(128)))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation("sigmoid"))

test_case = "Bidirectional_LSTM"
cluster_train_eval_strip(
    model, x_train, y_train, x_test, y_test, batch_size, test_case)
