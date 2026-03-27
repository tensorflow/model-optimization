## Limitations: Quantized Fused Activations in TFLite

When converting TensorFlow / Keras models to **quantized TFLite models**, not all
activation functions are currently supported as **fused activations** by the
TFLite MLIR converter.

### Unsupported Case: Add + Tanh Fusion

Although the TFLite schema defines `TANH` as a possible fused activation,
the current TFLite MLIR converter **does not fuse `Add` + `tanh`** during
quantized conversion.

As a result:

- `Add` and `tanh` are emitted as **separate operations**
- Quantized outputs may **differ from the original floating-point Keras model**
- This behavior is expected and represents a current limitation

At present, only the following activations are fused during quantized conversion:

- `RELU`
- `RELU6`
- `RELU_N1_TO_1`

### Minimal Reproducible Example

```python
import tensorflow as tf
import numpy as np

# Simple model with Add + tanh
inputs = tf.keras.Input(shape=(4,))
x = tf.keras.layers.Add()([inputs, inputs])
outputs = tf.keras.activations.tanh(x)
model = tf.keras.Model(inputs, outputs)

# Representative dataset for quantization
def representative_dataset():
    for _ in range(100):
        yield [np.random.rand(1, 4).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()


