# Model Weight Clustering Implementation for Model Optimization Toolkit

| Status        | Accepted                                             |
:---------------|:-----------------------------------------------------|
| **Author(s)** | Arm                                  |
| **Sponsor**   | Alan Chiao (TFMOT)                                |
| **Updated**   | 2020-01-30                                           |

## Objective

This proposal addresses the technique, and consequent API change, of weight clustering, an approach for compressing neural networks by restricting the number of unique weights so that a lower number of bits is required to represent each weight.

## Motivation

Clustering, or weight sharing, allows for certain types of hardware to benefit from advanced weight compression techniques and the associated reduction in model memory footprint and bandwidth.
Clustering can also be combined with pruning and quantization reducing the network’s storage requirements even further.

## Design Proposal

This document proposes and demonstrates the API changes for the clustering implementation.

### Implementation

The description of the implemented clustering technique that used as the basis for this proposal can be found at https://arxiv.org/abs/1510.00149

From a practical standpoint, the implementation is based on using a lookup table to hold the cluster centroid values during model training. The weight array is populated with a 'gather' operation so that during back propagation the gradients can be calculated in the normal way. The lookup table is then adjusted using the cumulative gradient values for the weights that correspond to the same centroid.

#### Proposed module structure

*This PR does not affect any of the existing pruning functionality and
is completely independent from the rest of the existing code base.*

* cluster.py/cluster_test.py - high level clustering interfaces and tests
* cluster_wrapper.py/cluster_wrapper_test.py - an implementation of a
particular clustering algorithm and associated tests
* clusterable_layer.py - similar to PrunableLayer, provides a wrapper for
a layer to be recognized as a clusterable one.
* clustering_centroids.py/clustering_centroids_test.py - implementations
of different clusters centroids initialization mechanics and tests for
them
* clustering_registry.py/clustering_registry_test.py - similar to
PruningRegistry contains the list of clusterable parameters in each
layer.


### Clustering API

The API is intended to be used in the same way the pruning API is used. The interface is intentionally made very similar to the one described in https://www.tensorflow.org/model_optimization/guide/pruning/.

The clustering API will reveal the following new interfaces to the end user, ```cluster_weights```, ```cluster_scope```, and ```strip_clustering```. The definition and function of which are demonstrated below.

#### Cluster Weights

```Cluster_weights``` is the main function to be used for clustering a model or a layer. It accepts a layer, a list of layers, or a Keras model to cluster, as well as a ```clustering_params``` dictionary that should specify the number of unique weights per layer (or ```number_of_clusters```) and the centroid initialization method, the way by which the initial clusters are specified.

````python
def cluster_weights(to_cluster, number_of_clusters, cluster_centroids_init, **kwargs):
  """Modify a keras layer or model to be clustered during training.

  This function wraps a keras model or layer with clustering functionality
  which clusters the layer's weights during training. For examples, using
  this with number_of_clusters equals 8 will ensure that each weight tensor has
  no more than 8 unique values.

  Before passing to the clustering API, a model should already be trained and show some acceptable performance
  on the testing/validation sets.

  The function accepts either a single keras layer
  (subclass of `keras.layers.Layer`), list of keras layers or a keras model
  (instance of `keras.models.Model`) and handles them appropriately.

  If it encounters a layer it does not know how to handle, it will throw an
  error. While clustering an entire model, even a single unknown layer would lead
  to an error.

  Cluster a model:

```python
  clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': 'density-based'
  }

  clustered_model = cluster_weights(original_model, **clustering_params)
```

  Cluster a layer:

```python
  clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': 'density-based'
  }

  model = keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      cluster_weights(layers.Dense(2, activation='tanh'), **clustering_params)
  ])
```

  Arguments:
      to_cluster: A single keras layer, list of keras layers, or a
        `tf.keras.Model` instance.
      number_of_clusters: the number of cluster centroids to form when clustering a layer/model.
        For example, if number_of_clusters=8 then only 8 unique values will be used in each weight array.
      cluster_centroids_init: how to initialize the cluster centroids.
        Can have following values:
          1. 'random' : centroids are sampled using the uniform distribution between the minimum and maximum weight
          values in a given layer
          2. 'density-based' : density-based sampling. First, cumulative distribution function is built for weights,
          then y-axis is evenly spaced into number_of_clusters regions. After this the corresponding x values are
          obtained and used to initialize clusters centroids.
          3. 'linear' : cluster centroids are evenly spaced between the minimum and maximum values of a given weight
      **kwargs: Additional keyword arguments to be passed to the keras layer.
        Ignored when to_cluster is not a keras layer.

  Returns:
    Layer or model modified to include clustering related metadata.

  Raises:
    ValueError: if the keras layer is unsupported, or the keras model contains
    an unsupported layer.
  """
````

#### Cluster Scope

````python
def cluster_scope():
  """Provides a scope in which Clustered layers and models can be deserialized.

  If a keras model or layer has been clustered, it needs to be within this scope
  to be successfully deserialized.

  Returns:
      Object of type `CustomObjectScope` with clustering objects included.

  Example:

  ```python
  clustered_model = cluster_weights(model, **self.params)
  keras.models.save_model(clustered_model, keras_file)

  with cluster_scope():
    loaded_model = keras.models.load_model(keras_file)
  ```
  """
````

#### Strip Clustering

````python
def strip_clustering(model):
  """Strip clustering wrappers from the model.

  Once a model has been clustered, this method can be used
  to restore the original model with the clustered weights.

  Only sequential and functional models are supported for now.

  Arguments:
      model: A `tf.keras.Model` instance with clustered layers.

  Returns:
    A keras model with clustering wrappers removed.

  Raises:
    ValueError: if the model is not a `tf.keras.Model` instance.
    NotImplementedError: if the model is a subclass model.

  Usage:

  ```python
  orig_model = tf.keras.Model(inputs, outputs)
  clustered_model = cluster_weights(orig_model)
  exported_model = strip_clustering(clustered_model)
  ```
  The exported_model and the orig_model have the same structure.
  """
````


## End-to-end Example

Finally, an example on how to use the API to cluster and save a model is provided in the examples directory, under clustering.

## Future Roadmap

* Sparsity-aware clustering, to ensure the sparsity of pruned models is not destroyed during the clustering and fine-tuning process.
* Clustering schedule and callback, to provide more advanced training-time control of the clustering behaviour and progressive layer clustering throughout training.
* More to be added . . .