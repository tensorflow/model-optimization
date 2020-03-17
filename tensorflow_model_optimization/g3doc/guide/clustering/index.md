# Weight Clustering

This document provides an overview on weight clustering to help you determine how it fits with your use case. To dive right into the code, see the [Clustering with Keras](clustering_with_keras.ipynb) tutorial and the [API docs](../../api_docs/python). For additional details on how to use the Keras API, a deep dive into weight clustering, and documentation on more advanced usage patterns, see the [API usage guide](train_clustered_models.md).

## Overview

Clustering, or weight sharing, is a technique for reducing the number of unique weight values in a model. It first groups the weights of each layer into *N* clusters, then shares the cluster's centroid value for all the weights belonging to the cluster.

This technique brings improvements in terms of model compression. It reduces the number of bits used to represent the weights, thus saving memory bandwidth. This can be critical for deploying deep learning models on embedded systems with limited resources.

We've seen up to 5x improvements in model compression with minimal loss of accuracy, as demonstrated by the [results](#results) presented below.

### API Compatibility Matrix

Users can apply clustering with the following APIs:

*   Model building: `tf.keras` with only Sequential and Functional models
*   TensorFlow versions: TF 1.x for versions 1.14+ and 2.x.
    *   `tf.compat.v1` with a TF 2.X package and `tf.compat.v2` with a TF 1.X
        package are not supported.
*   TensorFlow execution mode: both graph and eager
*   Distributed training: `tf.distribute` with only graph execution

## Results

### Image Classification

<table>
  <tr>
    <th rowspan=2>Model</th>
    <th colspan=2>Original</th>
    <th colspan=3>Clustered</th>
  </tr>
  <tr>
    <th>Top-1 accuracy</th>
    <th>Size of compressed .tflite</th>
    <th># of clusters</th>
    <th>Top-1 accuracy</th>
    <th>Size of compressed .tflite</th>
  </tr>
  <tr>
    <td>MobileNetV2</td>
    <td>72.29%</td>
    <td>13.0 MB</td>
    <td>32</td>
    <td>69.33%</td>
    <td>2.6 MB</td>
  </tr>
</table>

The models were trained and tested on ImageNet.

### Keyword Spotting

<table>
  <tr>
    <th rowspan=2>Model</th>
    <th colspan=2>Original</th>
    <th colspan=3>Clustered</th>
  </tr>
  <tr>
    <th>Top-1 accuracy</th>
    <th>Size of compressed .tflite</th>
    <th># of clusters</th>
    <th>Top-1 accuracy</th>
    <th>Size of compressed .tflite</th>
  </tr>
  <tr>
    <td>DS-CNN-L</td>
    <td>95.03%</td>
    <td>1.5 MB</td>
    <td>32</td>
    <td>94.71%</td>
    <td>0.3 MB</td>
  </tr>
</table>

The models were trained and tested on SpeechCommands v0.02.

NOTE: *Size of compressed .tflite* refers to the size of the zipped .tflite file obtained from the model through the following process:
1. Serialize the Keras model into .h5 file
2. Convert the .h5 file into .tflite using `TFLiteConverter.from_keras_model_file()`
3. Compress the .tflite file into a zip

## Examples

In addition to the [Clustering with Keras](clustering_with_keras.ipynb) tutorial, see the following examples:

* Cluster the weights of a CNN model trained on the MNIST handwritten digit classification databaset:
[code](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/examples/clustering/keras/mnist/mnist_cnn.py)

## Tips

1. The current clustering API works only with pre-trained models. Don't forget to train your model before attempting to cluster it.
2. The centroid initialization technique you opt for has a significant impact on the accuracy of the clustered model. Experiments have shown that linear initialization outperforms density-based and random initialization in most cases.

## References

The weight clustering implementation is based on the technique described in chapter 3, entitled *Trained Quantization and Weight Sharing*, of the conference paper referenced below.

1.  **Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding** <br/>
    Song Han, Huizi Mao, William J. Dally <br/>
    [https://arxiv.org/abs/1510.00149](https://arxiv.org/abs/1510.00149). ICLR, 2016 <br/>
