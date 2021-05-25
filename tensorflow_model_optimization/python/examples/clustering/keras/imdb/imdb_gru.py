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
"""Train a GRU on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF+LogReg.
"""

from __future__ import print_function

import tensorflow.keras as keras
import tensorflow.keras.preprocessing.sequence as sequence
from tensorflow_model_optimization.python.core.clustering.keras import cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster_config


max_features = 20000
maxlen = 100  # cut texts after this number of words
batch_size = 32

print("Loading data...")
(x_train,
 y_train), (x_test,
            y_test) = keras.datasets.imdb.load_data(num_words=max_features)
print(len(x_train), "train sequences")
print(len(x_test), "test sequences")

print("Pad sequences (samples x time)")
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

print("Build model...")
model = keras.models.Sequential()

model.add(keras.layers.Embedding(max_features, 128, input_length=maxlen))
model.add(keras.layers.GRU(128))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation("sigmoid"))

model = cluster.cluster_weights(
    model,
    number_of_clusters=16,
    cluster_centroids_init=cluster_config.CentroidInitialization
    .KMEANS_PLUS_PLUS,
)

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])


print("Train...")
model.fit(x_train, y_train, batch_size=batch_size, epochs=3,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)

print("Test score:", score)
print("Test accuracy:", acc)
