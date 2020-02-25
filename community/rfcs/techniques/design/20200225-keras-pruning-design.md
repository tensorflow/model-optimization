# Keras Pruning API Design

Status        | Proposed
:------------ | :-------
**RFC #**     | [NNN](https://github.com/tensorflow/model-optimization/pull/NNN) (update when you have the PR #)
**Author(s)** | Sample Name (me@email.com)
**Sponsor**   | Sample Sponsor (whomever@email.com)
**Updated**   | 2020-02-25

Preface to be moved to CONTRIBUTING_TECHNIQUE.md.

The basis for this comes from
https://github.com/tensorflow/community/blob/master/governance/TF-RFCs.md.

*   The proposal RFC is the "it is a good idea to discuss your aims with project
    contributors and maintainers and get early feedback" part. Given that model
    optimization is a specific domain, we formalize this step via the proposal
    RFC. Once accepted, the authors can proceed to design and implement the
    technique with the goal of launching the tool.

*   The design RFC is the same as the TF RFC doc with these specific aspects:

    1.  Comparison with existing APIs in TFMOT to stay consistent when it makes
        sense for the end-user. Once accepted, the API is the state where it
        should be at launch.
        *   TODO: "After writing the RFC draft, get feedback from these experts
            before submitting it." : how does that happen? Google Docs?
    2.  More standardized aspects, including the set of Tutorials & Examples
        content to consider (with varying priorities), performance testing, and
        the maintainer.
    3.  Slight reordering of sections to prioritize what affects the end-user
        first and what is most often different between techniques.

--------------------------------------------------------------------------------

<!-- This should be replaced with a link to the technique's overview page after launch. -->

See the initial proposal
[here](https://github.com/tensorflow/model-optimization/pull/NNN) for the
objective, motivation, user benefit and impact for the technique.

## Design Proposal

### Public APIs

#### Summary of public APIs

All APIs are exposed in the tfmot.sparsity.keras package (e.g.
tfmot.sparsity.keras.prune_low_magnitude). The following APIs are available for
different aspects of using pruning.

1.  Define the model:

    *   prune_low_magnitude: to prune a model or subset of layers.
    *   PrunableLayer: to prune custom Keras layers or prune different weights
        from API defaults.

2.  Train the model:

    *   PruningSchedule: to vary the degree of pruning based on the training
        step.
        *   ConstantSparsity: a PruningSchedule where the pruning sparsity is
            constant over a period of time.
        *   PolynomialDecay: a PruningSchedule where the pruning rate increases
            rapidly at first and then slows down to reach the target sparsity.
    *   PruningSummaries: log information including sparsity % to Tensorboard
    *   UpdatePruningStep: update the step, which is necessary for the
        PruningSchedule algorithms.

3.  Checkpoint and Deserialization

    *   prune_scope: deserialization of Keras hdf5 models. This is standard
        Keras.

4.  Deployment

    *   strip_pruning: for exporting model in a manner where compression gains
        are realized. All parts of pruning that are unneeded for deployment are
        removed.

#### Public APIs

See Tutorials & Examples section for examples of how these APIs are used.

##### Define the model

```
def prune_low_magnitude(to_prune,
                        pruning_schedule=pruning_sched.ConstantSparsity(0.5, 0),
                        block_size=(1, 1),
                        block_pooling_type='AVG',
                        **kwargs):
  """Modify a tf.keras layer or model to be pruned during training.

  Arguments:
    to_prune: A single keras layer, list of keras layers, or a
      `tf.keras.Model` instance.
    pruning_schedule: A `PruningSchedule` object that controls pruning rate
      throughout training.
    block_size: (optional) The dimensions (height, weight) for the block
      sparse pattern in rank-2 weight tensors.
    block_pooling_type: (optional) The function to use to pool weights in the
      block. Must be 'AVG' or 'MAX'.
    **kwargs: Additional keyword arguments to be passed to the keras layer.
      Ignored when to_prune is not a keras layer.

  Returns:
    Layer or model modified with pruning wrappers. Optimizer is removed.

  Raises:
    ValueError: if the keras layer is unsupported, or the keras model contains
    an unsupported layer.

  """
```

```
class PrunableLayer(object):
  """Abstract Base Class for making your own keras layer prunable.

  Custom keras layers which want to add pruning should implement this class.

  """

  @abc.abstractmethod
  def get_prunable_weights(self):
    """Returns list of prunable weight tensors.

    All the weight tensors which the layer wants to be pruned during
    training must be returned by this method.

    Returns: List of weight tensors/kernels in the keras layer which must be
        pruned during training.
    """
    raise NotImplementedError('Must be implemented in subclasses.')
```

##### Deployment

```
def strip_pruning(model):
  """Remove parts of pruning that are unnecessary for deployment.

  This is needed to see compression benefits for deployment.

  Once a model has been pruned to required sparsity, this method can be used
  to restore the original model with the sparse weights.

  Only sequential and functional models are supported for now.

  Arguments:
      model: A `tf.keras.Model` instance with pruned layers.

  Returns:
    A keras model with pruning wrappers removed.

  Raises:
    ValueError: if the model is not a `tf.keras.Model` instance.
    NotImplementedError: if the model is a subclass model.
  """
```

### Tutorials & Examples

To map different use cases to their relevant APIs.

*   Without a pruned model, you must **define** and **train** the model.
*   For Keras HDF5 models only, you need special **checkpointing and
    deserialization**. Checkpointing cannot be done with Keras HDF5 weights.
*   For deployment only, you must take steps to see compression benefits.

#### Define a model

##### Prune all layers in Functional and Sequential models

**Tips** for better model accuracy:

*   Try "Prune subset of layers" to skip pruning the layers that affect accuracy
    the most.
*   Generally better to start from pre-trained weights.

```
model = setup_model()
model.load_weights(pretrained_weights) # optional but recommended.

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model)
```

##### Prune subset of layers in Functional and Sequential models

**Tips** for better model accuracy:

*   Generally better to start from pre-trained weights.
*   Try pruning the later layers instead of the first layers.
*   Avoid pruning critical layers (e.g. attention mechanism).

```
model = setup_model()
model.load_weights(pretrained_weights) # optional but recommended

def layers_to_prune():
  # Knowing that the first layer is the Dense layer.
  return {model.layers[0]: 'default'}

def apply_pruning(layer):
  if layer in layers_to_prune():
    return tfmot.sparsity.keras.prune_low_magnitude(layer)
  return layer

pruned_model = tf.keras.models.clone_model(
    model,
    clone_function=apply_pruning,
)
```

###### More readable but potentially less accurate

This is not compatible with using pre-trained weights, which is why it may be
less accurate.

Functional example

```
i = tf.keras.Input(shape=(20,))
x = tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(10))(i)
o = tf.keras.layers.Flatten()(x)

pruned_model = tf.keras.Model(inputs=i, outputs=o)

```

Sequential example

```
pruned_model = tf.keras.Sequential([
  tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(20, input_shape=input_shape)),
  tf.keras.layers.Flatten()
])
```

#### Prune layers in Subclassed model

**Note**: using pre-trained weights is not supported yet.

**Tips** for better model accuracy:

*   Trying pruning the later layers instead of the first layers
*   Avoid pruning critical layers (e.g. attention mechanism).

```
class MyPrunedModel(tf.keras.Model):
  def __init__(self):
    super(MyPrunedModel, self).__init__()
    self.dense = tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(10))
    self.flatten = tf.keras.layers.Flatten()
    self.dense2 = tfmot.sparsity.keras.prune_low_magnitude(
        tf.keras.Sequential([tf.keras.layers.Dense(10)])
    )

  def call(self, inputs):
    x = self.dense(inputs)
    x = self.flatten(x)
    return self.dense2(x)

pruned_model = MyPrunedModel()

input_shape = (None, 20)
pruned_model.build(input_shape)

```

#### Prune custom Keras layer or prune different weights from API default

**Common mistake:** pruning the bias usually harms model accuracy too much.

```
class MyDenseLayer(tf.keras.layers.Dense, tfmot.sparsity.keras.PrunableLayer):

  def get_prunable_weights(self):
    # Prune bias also, though that usually harms model accuracy too much.
    return [self.kernel, self.bias]

class MyDenseLayer2(tf.keras.layers.Dense, tfmot.sparsity.keras.PrunableLayer):

  def get_prunable_weights(self):
    # Prune nothing.
    return []
```

### Train model

#### Model.fit

```
model = setup_model()
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model)

log_dir = tempfile.mkdtemp()
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    # Log sparsity and other metrics in Tensorboard.
    tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
]

pruned_model.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer='adam',
      metrics=['accuracy']
)

pruned_model.fit(
    x_train,
    y_train,
    callbacks=callbacks
)

```

#### Custom training loop

```
model = setup_model()
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model)

# Boilerplate
loss = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam()
log_dir = tempfile.mkdtemp()
unused_arg = -1
epochs = 1
batches = 1 # example is hardcoded so that the number of batches cannot change.

# Non-boilerplate.
pruned_model.optimizer = optimizer
step_callback = tfmot.sparsity.keras.UpdatePruningStep()
step_callback.set_model(pruned_model)
log_callback = tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir) # Log sparsity and other metrics in Tensorboard.
log_callback.set_model(pruned_model)

step_callback.on_train_begin() # run pruning callback
for _ in range(epochs):
  for _ in range(batches):
    step_callback.on_train_batch_begin(batch=unused_arg) # run pruning callback

    with tf.GradientTape() as tape:
      logits = pruned_model(x_train, training=True)
      loss_value = loss(y_train, logits)
      grads = tape.gradient(loss_value, pruned_model.trainable_variables)
      optimizer.apply_gradients(zip(grads, pruned_model.trainable_variables))

  step_callback.on_epoch_end(batch=unused_arg) # run pruning callback
  log_callback.on_epoch_end(batch=unused_arg) # run pruning callback

```

#### Improve pruned model accuracy

First, look at the
[prune_low_magnitude API docs](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/prune_low_magnitude)
to understand what a pruning schedule is and the math of each type of pruning
schedule.

**Tips**:

*   Have a learning rate that's not too high or too low when the model is
    pruning. Consider the
    [pruning schedule](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/PruningSchedule)
    to be a hyperparameter.

*   As a quick test, try running an experiment where you prune a model to the
    final sparsity with begin step 0 with a
    [ConstantSparsity](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/ConstantSparsity)
    schedule. You might get lucky with good results.

*   Do not prune very frequently to give the model time to recover. The
    [pruning schedule](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/PruningSchedule)
    provides a decent default frequency.

*   For general ideas to improve model accuracy, find your use case(s) under
    "Define model" on the navigation sidebar and see if there are tips.

### Checkpointing and deserialization

**Your Use Case:**

*   You cannot do checkpointing with Keras HDF5 weights since we need to
    preserve the step.

*   This code is only needed for the HDF5 model format (not HDF5 weights or
    other formats).

```
model = setup_model()
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model)

_, keras_model_file = tempfile.mkstemp('.h5')
# Saving the optimizer is necessary for checkpointing (True is the default).
pruned_model.save(keras_model_file, include_optimizer=True)

with tfmot.sparsity.keras.prune_scope():
  loaded_model = tf.keras.models.load_model(keras_model_file)

```

### Deployment

#### Export model with size compression

**Common mistake**: both `strip_pruning` and applying a standard compression
algorithm (e.g. via gzip) are necessary to see the compression benefits of
pruning.

```
# See "Define model" and "Train model" on navigation sidebar for how to define
# and train this model in other ways.
model = setup_model()
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model)

pruned_model.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer='adam',
      metrics=['accuracy']
)

pruned_model.fit(
    x_train,
    y_train,
    callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
)

final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
```

#### Hardware-specific optimizations

Once the framework
[enables pruning to improve latency](\(https://github.com/tensorflow/model-optimization/issues/173\)),
using block sparsity can improve latency for certain hardware. For a target
model accuracy, latency can still improve despite the fact that increasing the
block size will decrease the peak sparsity %.

```
model = setup_model()

# For using intrinsics on a CPU with 128-bit registers, together with 8-bit
# quantized weights, a 1x16 block size is nice because the block perfectly
# fits into the register.
pruning_params = {'block_size': [1, 16]}
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
```

<!-- See the technique overview page documentation template for what to consider
here. Includes potential support matrix section. -->

## Compatibility

### At launch

Users can apply pruning with the following APIs:

*   Model building: `tf.keras` with only Sequential and Functional models
*   TensorFlow versions: TF 1.x for versions 1.14+ and 2.x.
    *   `tf.compat.v1` with a TF 2.X package and `tf.compat.v2` with a TF 1.X
        package are not supported.
*   TensorFlow execution mode: both graph and eager
*   Distributed training: `tf.distribute` with only graph execution

### Roadmap

It is on our roadmap to add support in the following areas:

*   Minimal Subclassed model support, meaning that we automatically prune any
    weights that are created inside a Keras layer (whether builtin or custom
    PrunableLayer).

This is important because several critical use cases (e.g. Object Detection,
BERT) use Subclassed models today.

## Comparisons with existing TFMOT tools

1.  In contrast to the Keras quantization API, pruning only acts on a single
    Keras layer at a time, meaning there does not have to be a two-phase API for
    applying pruning (e.g. `prune_annotate` and `prune_apply`).

    *   This also enables pruning to have a PrunableLayer interface, as opposed
        to a QuantizeConfig interface. The former is friendlier for sharing the
        same configuration of the algorithm across the same layer when pruning a
        subset of layers. We will evaluate unifying these interfaces in the
        future.
        *   QuantizeConfig is needed for quantization for implementation
            reasons. Internally, the tool relies on the built-in Keras class
            type to detect graphs or sequences of layers to transform. Using a
            QuantizableLayer interface modifies the class type, which makes the
            detection more difficult.
    *   This also enables pruning to potentially have better Subclassed Model
        support for deployment because Subclassed models lack the graph of
        layers data structure needed for quantization to do cross-layer
        modifications.

2.  In contrast to weight clustering, pruning has been successfully used when
    training a model from scratch, as opposed to only starting from a
    pre-trained model.

### Performance Implications

As with other TFMOT techniques, there are training convergence and benchmark
tests to ensure no regresssion and that the tool continues to produce SOTA
results.

### Dependencies

As the first technique in the TFMOT package, this introduces the following
dependencies:

1.  numpy
2.  six : for Python 2 / 3 compatibility
3.  enum34 : for Python 2 / 3 compatibility
4.  tensorflow

### Engineering Impact

Expected impact on binary size, startup time, build time, and test times is
minimal. Build time is a few seconds on a 2013 laptop via bazel. Test time is ~N
minutes.

Maintenance is by the authors, in accordance to TFMOT's ownership principles.
TODO: link to ownership RFC when public.

### Platforms and Environments

For training, all platforms that Keras supports are supported. Our regular tests
would run on CPU, and our convergence and benchmark tests run on GPU and TPU.

For serving, see the [proposal doc](). For how the benefits vary across
platforms.

### Best Practices

N/A. This is the first technique in the TFMOT package.
