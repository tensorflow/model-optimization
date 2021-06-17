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
"""End-to-end tests for StackedRNNCells and PeepholeLSTMCell.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF+LogReg.
"""

from __future__ import print_function

import tensorflow.keras as keras
import tensorflow.keras.preprocessing.sequence as sequence
from tensorflow_model_optimization.python.core.clustering.keras import cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from imdb_utils import prepare_dataset, cluster_train_eval_strip


max_features = 20000
maxlen = 100  # cut texts after this number of words
batch_size = 32

x_train, y_train, x_test, y_test = prepare_dataset()

print("Build a model with the StackedRNNCells with PeepholeLSTMCell...")
model = keras.models.Sequential()

model.add(keras.layers.Embedding(max_features, 128, input_length=maxlen))
model.add(
  keras.layers.RNN(
    keras.layers.StackedRNNCells(
        [keras.experimental.PeepholeLSTMCell(128) for _ in range(2)])))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation("sigmoid"))

cluster_train_eval_strip(model, x_train, y_train, x_test, y_test, batch_size)

print("Build a model with the StackedRNNCells with LSTMCell...")
model = keras.models.Sequential()

model.add(keras.layers.Embedding(max_features, 128, input_length=maxlen))
model.add(
  keras.layers.RNN(
    keras.layers.StackedRNNCells(
        [keras.layers.LSTMCell(128) for _ in range(2)])))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation("sigmoid"))

cluster_train_eval_strip(model, x_train, y_train, x_test, y_test, batch_size)
