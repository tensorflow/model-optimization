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
"""Train a simple convnet with MultiHeadAttention layer on MNIST dataset
and cluster it.
"""
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import numpy as np

NUMBER_OF_CLUSTERS = 3

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# define model
input = tf.keras.layers.Input(shape=(28, 28))
x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=16, name="mha")(
    query=input, value=input
)
x = tf.keras.layers.Flatten()(x)
out = tf.keras.layers.Dense(10)(x)
model = tf.keras.Model(inputs=input, outputs=out)

# Train the digit classification model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    train_images, train_labels, epochs=1, validation_split=0.1,
)

score = model.evaluate(test_images, test_labels, verbose=0)
print('Model test loss:', score[0])
print('Model test accuracy:', score[1])

# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 1
validation_split = 0.1  # 10% of training set will be used for validation set.

# Define model for clustering
cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

clustering_params = {
    "number_of_clusters": NUMBER_OF_CLUSTERS,
    "cluster_centroids_init": CentroidInitialization.KMEANS_PLUS_PLUS,
}
model_for_clustering = cluster_weights(model, **clustering_params)

# `cluster_weights` requires a recompile.
model_for_clustering.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model_for_clustering.fit(
    train_images,
    train_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=validation_split,
)

score = model_for_clustering.evaluate(test_images, test_labels, verbose=0)
print('Clustered model test loss:', score[0])
print('Clustered model test accuracy:', score[1])

# Strip clustering from the model
clustered_model = tfmot.clustering.keras.strip_clustering(model_for_clustering)
clustered_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

score = clustered_model.evaluate(test_images, test_labels, verbose=0)
print('Stripped clustered model test loss:', score[0])
print('Stripped clustered model test accuracy:', score[1])

# Check that numbers of weights for MHA layer is the given number of clusters.
mha_weights = list(filter(lambda x: 'mha' in x.name and 'kernel' in x.name, clustered_model.weights))
for x in mha_weights:
    assert len(np.unique(x.numpy())) == NUMBER_OF_CLUSTERS
