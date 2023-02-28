This directory is modified based on default_8bit, which allows you to manually
change the number of bits of weight and activation in QAT.

Code example given a Keras float `model`:

```
# Imports.
import tensorflow_model_optimization as tfmot

from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_annotate_model
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_apply

from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit import default_n_bit_quantize_scheme


# TODO(user): define Keras float model.

# Specify scheme with 4-bit weights and 8-bit activations.
qat_scheme_4w8a = default_n_bit_quantize_scheme.DefaultNBitQuantizeScheme(
  num_bits_weight=4,
  num_bits_activation=8,
)

# Annotate the model for quantized aware training.
with tfmot.quantization.keras.quantize_scope():
  quantized_aware_model = quantize_apply(
    quantize_annotate_model(model),
    qat_scheme_4w8a,
  )

# TODO(user): compile and train quantized_aware_model using standard Keras methods.
```

The recommended activation precision is 8-bit for TF Lite conversion.

Before TF 2.11.0 the TF Lite converted weight value is stored one per byte in the weight tensor, so a 4-bit weight using default_n_bit scheme will be integer [-7, 7] occupying a byte.  With TF 2.11.0 and release candidate TF 2.12.0 weight packing for 4-bit weights is added for selected operators, so two 4-bit weights are packed per byte for the regular convolution operator in TF 2.11.0.

To improve task quality it may be necessary to specify higher weight precision for the first and last layers of the model such as 8-bit.  This can be achieved using wrapper code per layer.  A code example is shown in [kws_streaming](https://github.com/google-research/google-research/commit/c87bac8133e00dc4fe646c182072676146312e0f) framework in Google Research repository.
