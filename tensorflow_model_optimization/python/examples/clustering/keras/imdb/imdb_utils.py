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
"""Common utils for testing RNN e2e tests.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF+LogReg.
"""
from __future__ import print_function

import tensorflow.keras as keras
import tensorflow.keras.preprocessing.sequence as sequence
from tensorflow_model_optimization.python.core.clustering.keras import cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster_config


def prepare_dataset():
  max_features = 20000
  maxlen = 100  # cut texts after this number of words

  print("Loading data...")
  (x_train,
  y_train), (x_test,
             y_test) = keras.datasets.imdb.load_data(num_words=max_features)
  print(len(x_train), "train sequences")
  print(len(x_test), "test sequences")

  print("Pad sequences (samples x time)")
  x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
  x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

  return x_train, y_train, x_test, y_test

def cluster_train_eval_strip(
    model, x_train, y_train, x_test, y_test, batch_size):
  model = cluster.cluster_weights(
      model,
      number_of_clusters=16,
      cluster_centroids_init=cluster_config.CentroidInitialization
      .KMEANS_PLUS_PLUS,)

  model.compile(loss="binary_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])


  print("Train...")
  model.fit(x_train, y_train, batch_size=batch_size, epochs=1,
            validation_data=(x_test, y_test), verbose=2)
  score, acc = model.evaluate(x_test, y_test,
                              batch_size=batch_size)

  print("Test score:", score)
  print("Test accuracy:", acc)

  print("Strip clustering wrapper...")
  model = cluster.strip_clustering(model)
  layer_weight = getattr(model.layers[1].cell.cells[0], 'kernel')
  print("Number of clusters:", len(set(layer_weight.numpy().flatten())))
