# TensorFlow model optimization roadmap

**Updated: May 9th, 2019**

## Quantization
* Post training quantization for hybrid kernels -- [Launched](https://medium.com/tensorflow/introducing-the-model-optimization-toolkit-for-tensorflow-254aca1ba0a3)
* Post training quantization for (8b) fixed-point kernels
* Training with quantization for (8b) fixed-point kernels
* Post training quantization for reduced-float (16b) kernels
* Extend post and during training APIs to (8b) fixed-point RNNs
* Training with quantization for low bit-width (< 8b) fixed-point kernels

## Pruning / Sparsity
* Magnitude based weight pruning during training - Launched
* Support for sparse model execution

