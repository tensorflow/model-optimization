# TensorFlow model optimization roadmap

**Updated: May 14th, 2019**

## Quantization

*   Post training quantization for hybrid kernels --
    [Launched](https://medium.com/tensorflow/introducing-the-model-optimization-toolkit-for-tensorflow-254aca1ba0a3)
*   Post training quantization for (8b) fixed-point kernels --
    [Launched](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-post-training-integer-quantization-b4964a1ea9ba)
*   Quantization-aware Training for (8b) fixed-point kernels
*   Extend post and during training APIs to (8b) fixed-point RNNs
*   Quantization tooling for low bit-width (< 8b) fixed-point kernels

## Pruning / Sparsity
* Magnitude based weight pruning during training -- [Launched](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a)
* Support for sparse model execution

