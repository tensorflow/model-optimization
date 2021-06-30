**Updated: June, 2021**

TensorFlow’s Model Optimization Toolkit (MOT) has been used widely for
converting/optimizing TensorFlow models to TensorFlow Lite models with smaller
size, better performance and acceptable accuracy to run them on mobile and IoT
devices. We are now working to extend MOT techniques and tooling beyond
TensorFlow Lite to support TensorFlow SavedModel as well.

The following represents a high level overview of our roadmap. You should be
aware that this roadmap may change at any time and the order below does not
reflect any type of priority. We strongly encourage you to comment on our
roadmap and provide us feedback in the
[discussion group](https://groups.google.com/a/tensorflow.org/g/tflite).

## Quantization

#### TensorFlow Lite

*   Selective post-training quantization to exclude certain layers from
    quantization.
*   Quantization debugger to inspect quantization error losses per layer.
*   Applying quantization-aware training on more model coverage e.g. TensorFlow
    Model Garden.
*   Quality and performance improvements for post-training dynamic-range.
    quantization.

#### TensorFlow

*   Post Training Quantization (bf16 * int8 dynamic range).
*   Quantization Aware Training ((bf16 * int8 weight-only with fake quant).
*   Selective post-training quantization to exclude certain layers from
    quantization.
*   Quantization debugger to inspect quantization error losses per layer.

## Sparsity

#### TensorFlow Lite

*   Sparse model execution support for more models.
*   Target aware authoring for Sparsity.
*   Extend sparse op set with performant x86 kernels.

#### TensorFlow

*   Sparity support in TensorFlow.

## Cascading compression techniques

*   Quantization + Tensor Compression + Sparsity: demonstrate all
3 techniques working together.

## Compression

*   Tensor compression API to help compression algorithm developers implement
their own model compression algorithm (e.g. Weight Clustering) including
providing a standard way to test/benchmark.



