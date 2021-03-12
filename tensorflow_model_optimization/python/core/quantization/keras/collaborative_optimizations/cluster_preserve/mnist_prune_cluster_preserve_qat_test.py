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
"""Tests for a simple convnet with PCQAT on MNIST dataset. """
import numpy as np
import tensorflow as tf
import tempfile

from tensorflow_model_optimization.python.core.clustering.keras.experimental import cluster as exp_tfmot_cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster as tfmot_cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster_config as tfmot_cluster_config

from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras.collaborative_optimizations.cluster_preserve import (
    default_8bit_cluster_preserve_quantize_scheme,)
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule

from tensorflow_model_optimization.python.core.quantization.keras.collaborative_optimizations.cluster_preserve import cluster_utils


layers = tf.keras.layers
np.random.seed(1)
tf.random.set_seed(3)


def _build_model():
  """Create the baseline model."""
  i = tf.keras.layers.Input(shape=(28, 28), name='input')
  x = tf.keras.layers.Reshape((28, 28, 1))(i)
  x = tf.keras.layers.Conv2D(
      filters=12, kernel_size=(3, 3), activation='relu', name='conv1')(x)
  x = tf.keras.layers.MaxPool2D(2, 2)(x)
  x = tf.keras.layers.Flatten()(x)
  output = tf.keras.layers.Dense(10, name='fc2')(x)
  model = tf.keras.Model(inputs=[i], outputs=[output])

  return model


def _get_dataset():
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  # Use subset of 60000 examples to keep unit test speed fast.
  x_train = x_train[:1000]
  y_train = y_train[:1000]

  return (x_train, y_train), (x_test, y_test)


def _train_model(model, callback_to_use, num_of_epochs):
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'],)
  (x_train, y_train), _ = _get_dataset()
  model.fit(
      x_train,
      y_train,
      epochs=num_of_epochs,
      callbacks=callback_to_use,
      verbose=0)

  return model


def baseline_model():
  """Build, compile and train the baseline model."""
  base_model_epoch = 1
  model = _build_model()
  callbacks = []
  model = _train_model(model, callbacks, base_model_epoch)

  return model


def _prune_model(original_model):
  """ Apply the pruning wrapper, compile and train the model."""
  prune_epoch = 1
  pruning_params = {
      'pruning_schedule':
      pruning_schedule.ConstantSparsity(0.50, begin_step=0, frequency=10)
  }
  pruning_model = prune.prune_low_magnitude(original_model, **pruning_params)
  callbacks = [pruning_callbacks.UpdatePruningStep()]
  pruning_model = _train_model(pruning_model,
                               callbacks,
                               prune_epoch)
  pruning_model_stripped = prune.strip_pruning(pruning_model)

  return pruning_model, pruning_model_stripped


def _cluster_model(original_model, sparsity_flag):
  """ Apply the clustering wrapper, compile and train the model."""
  cluster_epoch = 1
  clustering_params = {
      'number_of_clusters': 8,
      'cluster_centroids_init':\
      tfmot_cluster_config.CentroidInitialization.DENSITY_BASED,
      'preserve_sparsity': sparsity_flag,
  }
  cluster_model = exp_tfmot_cluster.cluster_weights(
      original_model, **clustering_params
  )

  callbacks = []
  cluster_model = _train_model(cluster_model,
                               callbacks,
                               cluster_epoch)

  clustered_model_stripped = tfmot_cluster.strip_clustering(cluster_model)

  return cluster_model, clustered_model_stripped


def selective_cluster_model(original_model, sparsity_flag):
  cluster_epoch = 1
  clustering_params = {
      'number_of_clusters': 8,
      'cluster_centroids_init':\
      tfmot_cluster_config.CentroidInitialization.DENSITY_BASED,
      'preserve_sparsity': sparsity_flag,
  }

  def apply_clustering_to_conv2d(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
      return exp_tfmot_cluster.cluster_weights(layer, **clustering_params)
    return layer

  cluster_model = tf.keras.models.clone_model(
      original_model,
      clone_function=apply_clustering_to_conv2d,
  )

  callbacks = []
  cluster_model = _train_model(cluster_model,
                               callbacks,
                               cluster_epoch)

  clustered_model_stripped = tfmot_cluster.strip_clustering(cluster_model)

  return cluster_model, clustered_model_stripped


def prune_cluster_preserve_quantize_model(clustered_model, preserve_sparsity):
  """ Prune_cluster_preserve QAT model."""
  pcqat_epoch = 1
  quant_aware_annotate_model = \
      quantize.quantize_annotate_model(clustered_model)
  quant_aware_model = quantize.quantize_apply(
      quant_aware_annotate_model,
      scheme=default_8bit_cluster_preserve_quantize_scheme
      .Default8BitClusterPreserveQuantizeScheme(preserve_sparsity))

  callbacks = []
  quant_aware_model = _train_model(quant_aware_model,
                                   callbacks,
                                   pcqat_epoch)
  pcqat_stripped = cluster_utils.strip_clustering_cqat(quant_aware_model)

  return quant_aware_model, pcqat_stripped


def _get_num_unique_weights_kernel(model):
  """ Check Dense and Conv2D layers."""
  num_unique_weights_list = []
  for layer in model.layers:
    if isinstance(layer,
                  (tf.keras.layers.Conv2D,
                   tf.keras.layers.Dense,
                   quantize.quantize_wrapper.QuantizeWrapper)):
      for weights in layer.trainable_weights:
        if 'kernel' in weights.name:
          np_weights = tf.keras.backend.get_value(weights)
          unique_weights = len(np.unique(np_weights))
          num_unique_weights_list.append(unique_weights)

  return num_unique_weights_list


def _check_sparsity_kernel(model):
  sparsity_list = []
  for layer in model.layers:
    if isinstance(layer,
                  (prune.pruning_wrapper.PruneLowMagnitude,
                   quantize.quantize_wrapper.QuantizeWrapper)):
      for weights in layer.trainable_weights:
        if 'kernel' in weights.name:
          np_weights = tf.keras.backend.get_value(weights)
          sparsity = 1.0 - np.count_nonzero(np_weights) / float(
              np_weights.size)
          sparsity_list.append(sparsity)

  return sparsity_list


# This code is directly from
# https://www.tensorflow.org/model_optimization/guide/quantization/training_example
def _evaluate_model(interpreter, test_images, test_labels):
  input_index = interpreter.get_input_details()[0]['index']
  output_index = interpreter.get_output_details()[0]['index']

  # Run predictions on every image in the "test" dataset.
  test_images = test_images[:300]
  test_labels = test_labels[:300]

  prediction_digits = []
  for _, test_image in enumerate(test_images):
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)
    # Run inference.
    interpreter.invoke()
    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  # Compare prediction results with ground truth labels to calculate accuracy.
  prediction_digits = np.array(prediction_digits)
  accuracy = (prediction_digits == test_labels).mean()

  return accuracy


def _check_tflite_conversion(model, integer_only_quant):
  # This function is for checking the tflite model conversion and accuracy of
  # tflite evaluation.
  (x_train, _), (x_test, y_test) = _get_dataset()
  def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(
        x_train).batch(1).take(100):
      yield [input_value]

  if integer_only_quant:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model_quant = converter.convert()
  else:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_quant = converter.convert()

  interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
  interpreter.allocate_tensors()
  test_accuracy = _evaluate_model(interpreter, x_test, y_test)

  return test_accuracy


class FunctionalTest(tf.test.TestCase):
  # The input of PCQAT is prune-preserving clustered
  # model (full model clustered)
  def testPCQATTrainingE2E(self):
    # Baseline model training
    model = baseline_model()

    # Prune the model to have 50% sparsity
    pruned_model, pruned_stripped = _prune_model(model)
    sparsity_pruning = _check_sparsity_kernel(pruned_model)

    # Cluster the model with 8 clusters
    preserve_sparsity = True
    clustered_model, clustered_model_stripped = _cluster_model(
        pruned_stripped, preserve_sparsity)
    _, (x_test, y_test) = _get_dataset()
    pc_result = clustered_model.evaluate(x_test, y_test)
    self.assertGreater(pc_result[1], 0.8)

    # Case 1: apply PCQAT to the pruned_clustered_model with preserve_sparsity
    # flag on
    pcqat_model, pcqat_model_stripped = prune_cluster_preserve_quantize_model(
        clustered_model_stripped, True)

    # Evaluate the fp32 result of PCQAT
    pcqat_result = pcqat_model.evaluate(x_test, y_test)
    self.assertGreater(pcqat_result[1], 0.8)

    # Check the unique weights of clustered_model and pcqat_model
    num_of_unique_weights_clst = _get_num_unique_weights_kernel(
        clustered_model_stripped)
    num_of_unique_weights_pcqat = _get_num_unique_weights_kernel(
        pcqat_model_stripped)
    self.assertAllEqual(num_of_unique_weights_clst,
                        num_of_unique_weights_pcqat)
    # Check number of unique weights after clustering and pcqat should be
    # less or equal to 8
    self.assertAllLessEqual(num_of_unique_weights_clst, 8)

    # Compare sparsity per layer for pruned_model and pcqat_model
    sparsity_pcqat = _check_sparsity_kernel(pcqat_model)
    # The sparsity in pcqat should be greater or euqal to the original
    # uniform sparsity per layer. In this example, 0.5
    self.assertAllGreaterEqual(np.array(sparsity_pcqat),
                               sparsity_pruning[0])

    # Verify the tflite conversion and accuracy
    tflite_accuracy = _check_tflite_conversion(pcqat_model_stripped,
                                               False)
    self.assertGreater(tflite_accuracy, 0.8)

    # Check the accuracy of PCQAT is not worse than the one of PC
    self.assertGreaterEqual(pcqat_result[1], pc_result[1])

    # Case 2: when the preserve_sparsity flag is False, the final sparsity
    # of pcqat should be destroyed
    pcqat_model, pcqat_model_stripped = prune_cluster_preserve_quantize_model(
        clustered_model_stripped, False)
    # Check unique weights
    num_of_unique_weights_pcqat = _get_num_unique_weights_kernel(
        pcqat_model_stripped)
    self.assertAllEqual(num_of_unique_weights_clst,
                        num_of_unique_weights_pcqat)

    # Compare sparsity per layer for pruned_model and pcqat_model
    sparsity_pcqat = _check_sparsity_kernel(pcqat_model)
    # The sparsity in pcqat should be less than the original
    # uniform sparsity per layer. In this example, 0.5
    self.assertAllLess(np.array(sparsity_pcqat), sparsity_pruning[0])

  def testPCQATSelectiveClustering(self):
    # Baseline model training
    model = baseline_model()

    # Prune the model to have 50% sparsity
    pruned_model, pruned_stripped = _prune_model(model)
    sparsity_pruning = _check_sparsity_kernel(pruned_model)

    # Cluster the conv2d layers with 8 clusters
    preserve_sparsity = True
    clustered_model, clustered_model_stripped = selective_cluster_model(
        pruned_stripped, preserve_sparsity)
    _, (x_test, y_test) = _get_dataset()
    pc_result = clustered_model.evaluate(x_test, y_test)
    self.assertGreater(pc_result[1], 0.8)

    # Apply PCQAT on the pruned_clustered_model
    pcqat_model, pcqat_model_stripped = prune_cluster_preserve_quantize_model(
        clustered_model_stripped, True)

    # Evaluate the fp32 result of PCQAT
    pcqat_result = pcqat_model.evaluate(x_test, y_test)
    self.assertGreater(pcqat_result[1], 0.8)

    num_of_unique_weights_clst = _get_num_unique_weights_kernel(
        clustered_model_stripped)
    num_of_unique_weights_pcqat = _get_num_unique_weights_kernel(
        pcqat_model_stripped)
    # Selective clustering only applies to the conv2d layers,
    # here we only check the first layer's unique weights
    self.assertEqual(num_of_unique_weights_clst[0],
                     num_of_unique_weights_pcqat[0])

    # Check number of unique weights after clustering and pcqat should be
    # both less or equal to 8
    self.assertLessEqual(num_of_unique_weights_clst[0], 8)

    # Check sparsity for the selected layer for pruned_model and pcqat_model
    # to prove pcqat works on selective cases
    sparsity_pcqat = _check_sparsity_kernel(pcqat_model)
    self.assertGreaterEqual(sparsity_pcqat[0],
                            sparsity_pruning[0])

    # Check the tflite conversion for pcqat
    pcqat_tflite_accuracy = _check_tflite_conversion(pcqat_model_stripped,
                                                     False)
    self.assertGreater(pcqat_tflite_accuracy, 0.8)

    # Check the accuracy of PCQAT is not worse than the one of PC
    self.assertGreaterEqual(pcqat_result[1], pc_result[1])


if __name__ == '__main__':
  tf.test.main()
