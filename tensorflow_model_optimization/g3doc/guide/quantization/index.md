# Post-training quantization

Post-training quantization is a general technique to reduce model size while also
providing up to 3x lower latency with little degradation in model accuracy. Post-training
quantization quantizes weights from floating point to 8-bits of precision. This technique
is enabled as an option in the
[TensorFlow Lite converter](https://www.tensorflow.org/lite/convert/):

```
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
```

At inference, weights are converted from 8-bits of precision to floating point and
computed using floating-point kernels. This conversion is done once and cached to reduce latency.

For more information, see the TensorFlow Lite
[post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
guide.
