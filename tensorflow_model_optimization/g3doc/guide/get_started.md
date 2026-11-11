# Get started with TensorFlow model optimization

## 1. Choose the best model for the task

Depending on the task, you will need to make a tradeoff between model complexity
and size. If your task requires high accuracy, then you may need a large and
complex model. For tasks that require less precision, it is better to use a
smaller model because they not only use less disk space and memory, but they are
also generally faster and more energy efficient.

## 2. Pre-optimized models

See if any existing
[TensorFlow Lite pre-optimized models](https://www.tensorflow.org/lite/models)
provide the efficiency required by your application.

## 3. Post-training tooling

If you cannot use a pre-trained model for your application, convert your
already-trained model to a float32 LiteRT model and quantize it with
[LiteRT Quantizer](https://github.com/google-ai-edge/LiteRT/tree/main/litert_quantizer).

Note: Post-training quantization in the TensorFlow Lite converter is
deprecated. See the
[migration guide](https://developers.google.com/edge/litert/quantization/tflq_to_litertq_migration).

## Next steps: Training-time tooling

If the above simple solutions don't satisfy your needs, you may need to involve
training-time optimization techniques.
[Optimize further](optimize_further.md) with our training-time tools and dig
deeper.
