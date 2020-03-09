# Quantization-aware training

<!-- TODO(tfmot): finalize on discussion of navigation vs overview sentence first. -->
There are two forms of quantization: post-training quantization and
quantization-aware training. We recommend starting with [the first](index.md)
since it's easier to use, though the second is often better for model accuracy.

This page provides an overview on quantization-aware training to help you
determine how it fits with your use case.

<!-- TODO(tfmot): fix urls once ready -->

*   To dive right into an end-to-end example, see the
    [quantization-aware training with Keras example](quantization_aware_training.ipynb).
*   To quickly find the APIs you need for your use case, see the
    [quantization-aware training comprehensive guide](quantization_aware_training_guide.md).

### Overview

Quantization-aware training emulates inference-time quantization, creating a
model that downstream tools will use to produce actually quantized models.
The quantized models use lower-precision (e.g. 8-bit instead of 32-bit float),
leading to benefits during deployment.

#### Deploy with quantization

Quantization brings improvements via model compression and latency reduction.
With the API defaults, the model size shrinks by 4x, and we typically see
between 1.5 - 4x improvements in CPU latency in the tested backends. Further
latency improvements can be seen on compatible machine learning accelerators,
such as the [EdgeTPU](https://coral.ai/docs/edgetpu/benchmarks/) and NNAPI.

The technique is used in production in speech, vision, text, and translate use
cases. The code currently supports vision use cases and will expand over time.

#### Research quantization and associated hardware

Users can configure the quantization parameters (e.g. number of
bits) and to some degree, the underlying algorithms. With these changes
from the API defaults, there is no easy path to deployment.

#### API compatibility

Users can apply quantization with the following APIs:

*   Model building: `tf.keras` with only Sequential and Functional models.
*   TensorFlow versions: TF 2.x for tf-nightly
    *   `tf.compat.v1` with a TF 2.X package is not supported.
*   TensorFlow execution mode: eager execution

It is on our roadmap to add support in the following areas:

<!-- TODO(tfmot): file Github issues. -->

*   Model building: clarify how Subclassed Models have limited to no support
*   Distributed training: `tf.distribute`

#### General support matrix

Support is available in the following areas:

<!-- TODO(tfmot): link to layers when ready -->

*   Model coverage: Mobilenet v1 and v2 and models using whitelisted layers.
<!-- TODO(tfmot): add more details and ensure they are all correct. -->
*   Hardware acceleration: our API defaults are compatible with acceleration on
    EdgeTPU, NNAPI, and TFLite backends, amongst others.

It is on our roadmap to add support in the following areas:

<!-- TODO(tfmot): file Github issue. Update as more functionality is added prior
to launch. -->

*   Model coverage: extended coverage for vision and other use cases, including
    concat support.

### Results

#### Image classification with tools

<!-- TODO(tfmot): update numbers if new and old experiments have varying
results -->

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Non-quantized Top-1 Accuracy </th>
      <th>Quantized Accuracy </th>
    </tr>
    <tr>
      <td>MobilenetV1 224</td>
      <td>71.02%</td>
      <td>71.05%</td>
    </tr>
    <tr>
      <td>MobilenetV2 224</td>
      <td>71.9%</td>
      <td>71.1%</td>
    </tr>
 </table>
</figure>

The models were tested on Imagenet and evaluated in both TensorFlow and TFLite.

#### Image classification for technique

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Non-quantized Top-1 Accuracy </th>
      <th>Quantized Accuracy </th>
    <tr>
      <td>Nasnet-Mobile</td>
      <td>74%</td>
      <td>73%</td>
    </tr>
    <tr>
      <td>Resnet-v1 50</td>
      <td>75.2%</td>
      <td>75%</td>
    </tr>
    <tr>
      <td>Resnet-v2 50</td>
      <td>75.6%</td>
      <td>75%</td>
    </tr>
 </table>
</figure>

The models were tested on Imagenet and evaluated in both TensorFlow and TFLite.

### Examples

In addition to the
[quantization-aware training with Keras example](quantization_with_keras.ipynb),
see the following examples:

*   Train a CNN model on the MNIST handwritten digit classification task with
    quantization:
    [code](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/examples/quantization/keras/mnist_cnn.py)

For background on something similar, see the *Quantization and Training of
Neural Networks for Efficient Integer-Arithmetic-Only Inference*
[paper](https://arxiv.org/abs/1712.05877). This paper introduces some concepts
that this tool uses. The implementation is not exactly the same, and there are
additional concepts used in this tool (e.g. per-axis quantization).
