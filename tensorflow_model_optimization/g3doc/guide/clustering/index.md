# Weight clustering

<sub>Maintained by Arm ML Tooling</sub>

This document provides an overview on weight clustering to help you determine how it fits with your use case.

- To dive right into an end-to-end example, see the [weight clustering example](clustering_example.ipynb).
- To quickly find the APIs you need for your use case, see the [weight clustering comprehensive guide](clustering_comprehensive_guide.ipynb).

## Overview

Clustering, or weight sharing, reduces the number of unique weight values in a model, leading to benefits for deployment. It first groups the weights of each layer into *N* clusters, then shares the cluster's centroid value for all the weights belonging to the cluster.

This technique brings improvements via model compression. Future framework support can unlock memory footprint improvements that can make a crucial difference for deploying deep learning models on embedded systems with limited resources.

We have experimented with clustering across vision and speech tasks. We've seen up to 5x improvements in model compression with minimal loss of accuracy, as demonstrated by the [results](#results) presented below.

Please note that clustering will provide reduced benefits for convolution and dense layers that precede a batch normalization layer, as well as in combination with per-axis post-training quantization.

### API compatibility matrix

Users can apply clustering with the following APIs:

*   Model building: `tf.keras` with only Sequential and Functional models
*   TensorFlow versions: TF 1.x for versions 1.14+ and 2.x.
    *   `tf.compat.v1` with a TF 2.X package and `tf.compat.v2` with a TF 1.X
        package are not supported.
*   TensorFlow execution mode: both graph and eager

## Results

### Image classification

<table>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="2">Original</th>
    <th colspan="4">Clustered</th>
  </tr>
  <tr>
    <th>Top-1 accuracy (%)</th>
    <th>Size of compressed .tflite (MB)</th>
    <th>Configuration</th>
    <th># of clusters</th>
    <th>Top-1 accuracy (%)</th>
    <th>Size of compressed .tflite (MB)</th>
  </tr>
  <tr>
    <td rowspan="3">MobileNetV1</td>
    <td rowspan="3">70.976</td>
    <td rowspan="3">14.97</td>
  </tr>
  <tr>
    <td>Selective (last 3 Conv2D layers)</td>
    <td>16, 16, 16</td>
    <td>70.294</td>
    <td>7.69</td>
  </tr>
  <tr>
    <td>Selective (last 3 Conv2D layers)</td>
    <td>32, 32, 32</td>
    <td>70.69</td>
    <td>8.22</td>
  </tr>
  <tr>
    <td>Full (all Conv2D layers)</td>
    <td>32</td>
    <td>69.4</td>
    <td>4.43</td>
  </tr>
  <tr>
    <td rowspan="3">MobileNetV2</td>
    <td rowspan="3">71.778</td>
    <td rowspan="3">12.38</td>
  </tr>
  <tr>
    <td>Selective (last 3 Conv2D layers)</td>
    <td>16, 16, 16</td>
    <td>70.742</td>
    <td>6.68</td>
 </tr>
 <tr>
    <td>Selective (last 3 Conv2D layers)</td>
    <td>32, 32, 32</td>
    <td>70.926</td>
    <td>7.03</td>
 </tr>
 <tr>
   <td>Full (all Conv2D layers)</td>
   <td>32</td>
   <td>69.744</td>
   <td>4.05</td>
  </tr>
</table>

The models were trained and tested on ImageNet.

### Keyword spotting

<table>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="2">Original</th>
    <th colspan="4">Clustered</th>
  </tr>
  <tr>
    <th>Top-1 accuracy (%)</th>
    <th>Size of compressed .tflite (MB)</th>
    <th>Configuration</th>
    <th># of clusters</th>
    <th>Top-1 accuracy (%)</th>
    <th>Size of compressed .tflite (MB)</th>
  </tr>
  <tr>
    <td rowspan="3">DS-CNN-L</td>
    <td rowspan="3">95.233</td>
    <td rowspan="3">1.46</td>
  </tr>
  <tr>
   <td>Full (all Conv2D layers)</td>
   <td>32</td>
   <td>95.09</td>
   <td>0.39</td>
  </tr>
  <tr>
   <td>Full (all Conv2D layers)</td>
   <td>8</td>
   <td>94.272</td>
   <td>0.27</td>
  </tr>
</table>

The model was trained and tested on SpeechCommands v0.02.

NOTE: *Size of compressed .tflite* refers to the size of the zipped .tflite file obtained from the model from the following process:
1. Serialize the Keras model into .h5 file
2. Convert the .h5 file into .tflite using `TFLiteConverter.from_keras_model_file()`
3. Compress the .tflite file into a zip

## Examples

In addition to the
[Weight clustering in Keras example](clustering_example.ipynb), see the
following examples:

* Cluster the weights of a CNN model trained on the MNIST handwritten digit classification dataset:
[code](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/examples/clustering/keras/mnist/mnist_cnn.py)

The weight clustering implementation is based on the *Deep Compression:
Compressing Deep Neural Networks With Pruning, Trained Quantization and Huffman
Coding* [paper](https://arxiv.org/abs/1510.00149). See chapter 3, titled
*Trained Quantization and Weight Sharing*.
