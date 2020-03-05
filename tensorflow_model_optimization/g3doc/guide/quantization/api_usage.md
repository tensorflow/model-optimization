
# Quantization Aware Training

[TOC]

## API Usage

The Quantize Aware Training API has been designed to be very simple to get
started, while also being flexible enough to allow high configurability of
model quantization. It allows users to emulate quantization loss in their
model during training, so that the model learns parameters resistant to
quantization loss and improves accuracy.

### Quantize an entire Keras model.

The simplest action a user may want to do is quantize the entire model with
the default configuration. This just works out of the box with a few lines
of code, and is the best option for users who simply want to deploy a
production model to TensorFlow Lite.

```python
import tensorflow_model_optimization as tfmo

# Works for Sequential/Functional models, not subclass models.
model = tf.keras.Sequential([
   ...
])
quantized_model = tfmo.quantization.keras.quantize_model(model)

quantized_model.compile(...)
quantized_model.fit(...)
```

### Quantize subset of model.

It is possible that a user may want to quantize only a subset of the model.
The rest of the model can still be run in pure floating point.

```python
import tensorflow_model_optimization as tfmo
quantize_annotate_layer = tfmo.quantization.keras.quantize_annotate_layer

# Works for Sequential/Functional models, not subclass models.
model = tf.keras.Sequential([
   ...
   # Only annotated layers will be quantized.
   quantize_annotate_layer(Conv2D()),
   quantize_annotate_layer(ReLU()),
   # Dense won't be quantized since it hasn't been annotated.
   Dense(),
   ...
])
quantized_model = tfmo.quantization.keras.quantize_apply(model)
```

This ensures that only part of the model runs as quantized. Users can
do this for the most performance sensitive parts of their model.
This feature is possible due to our new experimental TensorFlow Lite MLIR
converter.

### Use custom quantize config for a layer.

Quantizing a model with default parameters has the advantage that it is
fully supported on TensorFlow Lite when converted, and works great for
production use cases. However, researchers and ML engineers may often want to
experiment with different quantization schemes. In that case, the library
makes it very easy to configure the quantization parameters/algorithms
differently. Please note however, that these schemes would not be supported
during inference on the TensorFlow Lite backend.

```python
import tensorflow_model_optimization as tfmo

quantize_annotate_layer = tfmo.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmo.quantization.keras.quantize_annotate_model

LastValueQuantizer = tfmo.quantization.keras.quantizers.LastValueQuantizer
QuantizeConfig = tfmo.quantization.keras.QuantizeConfig


class ConvQuantizeConfig(QuantizeConfig):
    """Custom QuantizeConfig for Conv layer.

    By default, both the weights and activations in Conv are quantized with
    8 bits. In this custom config, activations are not quantized, and weights
    are quantized with 4 bits.

    The QuantizeConfig allows you to precisely choose,
      a. What to quantized in a layer, and
      b. How to quantize it, by choosing the quantizer
    """

    def get_weights_and_quantizers(self, layer):
      # Use 4-bits to quantized weights instead of 8.
      return [(layer.kernel, LastValueQuantizer(num_bits=4, per_axis=True))]
    
    def get_activations_and_quantizers(self, layer):
      # Don't quantize the activation.
      return []

    def get_output_quantizers(self, layer):
      return []

# Works for Sequential/Functional models, not subclass models.
model = quantize_annotate_model(tf.keras.Sequential([
   ...
   # Quantize Conv2D with custom config.
   quantize_annotate_layer(Conv2D(), ConvQuantizeConfig()),
   # Other layers get quantized with default config.
   ReLU(),
   Dense(),
   ...
]))
quantized_model = tfmo.quantization.keras.quantize_apply(model)
```

### Write your own Quantization algorithm

As an ML engineer/researcher or hardware designer you may want to use a
specialized quantization algorithm for your model or hardware. Quantization
Aware Training allows you an easy way to write your own algorithm by
implementing the `Quantizer` interface, and plugging it into layers using
`QuantizeConfig`.

This allows users to both easily prototype the accuracy loss/benefits of
different quantization schemes, and train accurate models which target these
different schemes.

```python
import tensorflow_model_optimization as tfmo

Quantizer = tfmo.quantization.keras.quantizers.Quantizer
QuantizeConfig = tfmo.quantization.keras.QuantizeConfig


class FixedRangeQuantizer(Quantizer):
  """Quantizer which has a fixed range between -1 and 1."""

  def build(self, tensor_shape, name, layer):
    # Create any variables needed by your quantizer.
    self.min_weight = layer.add_weight(
        name + '_min',
        initializer=keras.initializers.Constant(-1.0),
        trainable=False)
    self.max_weight = layer.add_weight(
        name + '_max',
        initializer=keras.initializers.Constant(1.0),
        trainable=False)

  def __call__(self, inputs, step, training, **kwargs):
    """Fictional quantizer."""
    return clip_value(inputs, self.min_weight, self.max_weight)


# This custom Quantizer can now be used in a QuantizeConfig as specified above.
class ConvQuantizeConfig(QuantizeConfig):
    """Custom QuantizeConfig for Conv layer."""

    def get_weights_and_quantizers(self, layer):
      # Use FixedRangeQuantizer instead of default Quantizer.
      return [(layer.kernel, FixedRangeQuantizer())]
    
    def get_activations_and_quantizers(self, layer):
      # Don't quantize the activation.
      return []

    def get_output_quantizers(self, layer):
      return []
```

### Quantize a custom layer

TensorFlow Keras users often write custom layers by inheriting
`tf.keras.layers.Layer`, which are used for specialized computations. The API
allows users to apply quantization to these layers. The process is the same as
using custom configuration for existing layers - the user just needs to create
a `QuantizeConfig` for the layer. It is also possible to not quantize these
layers.

```python
import tensorflow_model_optimization as tfmo

LastValueQuantizer = tfmo.quantization.keras.quantizers.LastValueQuantizer
QuantizeConfig = tfmo.quantization.keras.QuantizeConfig
NoOpQuantizeConfig = tfmo.quantization.keras.NoOpQuantizeConfig

class MyLayer(Layer):
  def call(self, inputs):
    # Do stuff

class MyLayerQuantizeConfig(QuantizeConfig):
    """QuantizeConfig for MyLayer."""

    def get_weights_and_quantizers(self, layer):
      return [(layer.kernel, LastValueQuantizer())]

    def get_activations_and_quantizers(self, layer):
      return []

    def get_output_quantizers(self, layer):
      return []


model = quantize_annotate_model(tf.keras.Sequential([
   # Quantize layer with MyLayerQuantizeConfig.
   quantize_annotate_layer(MyLayer(), MyLayerQuantizeConfig()),
   # NoOpQuantizeConfig implies layer won't be quantized.
   quantize_annotate_layer(MyLayer(), NoOpQuantizeConfig()),
]))
quantized_model = tfmo.quantization.keras.quantize_apply(model)
```
