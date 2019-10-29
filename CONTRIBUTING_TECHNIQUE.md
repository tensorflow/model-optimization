# Guidelines for Contributing a New Technique

The following are guidelines for contributing a complete technique into the
toolkit, for example, connection pruning.

We will be evolving these guidelines to make the process more effective, and
as we receive more contributions. For example, we are working on creating a
repository of training scripts for
different models and tasks aimed at simplifying technique validation
([issue](https://github.com/tensorflow/model-optimization/issues/133)),
making the project contributor-friendly
([issue](https://github.com/tensorflow/model-optimization/issues/131)), and
having reproducible results.


## Contribution Components

1. Please start by providing an [RFC](https://github.com/tensorflow/community/blob/master/governance/TF-RFCs.md) under [model-optimization/community/rfcs](https://github.com/tensorflow/model-optimization/blob/master/community/rfcs).
   Consider the following guidelines:
   * API and implementation should strive for similarity with existing
     techniques to provide the best user experience.
   * consider the end-to-end experience for the user of your technique.
   * be prepared for a potential design discussion.

2. Provide experimental results that demonstrate benefits to end-users across
   models and tasks. This is probably the main criteria for us to consider, so
   the stronger the validation the better. Some relevant aspects are:
   * for Keras APIs, we recommend the following test tasks (and
     hope to be adding more):
     * [BERT task](https://github.com/tensorflow/models/tree/master/official/nlp/bert)
     * [object detection](https://github.com/tensorflow/models/tree/master/research/object_detection)
   * results in combination with other techniques (e.g. post-training integer
     quantization).
   * results include not only accuracy but also deployment metrics (e.g. model,
     storage space, latency, memory, to mention a few).
   * reproducible results are best: e.g. provide hyperparameters with minimal
     scripts to reproduce results.
   * when possible, include trained models that showcase those benefits.

3. Documentation and tutorials:
   * overview page that requires minimal end-user domain knowledge. [Sample](https://www.tensorflow.org/model_optimization/guide/pruning)
     * TODO(tfmot): template
   * colab tutorial that covers the most common use cases and user
     journeys. [Sample](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras)
   * advanced documentation that may cover:
       * advanced use cases not in tutorial. [Sample](https://www.tensorflow.org/model_optimization/guide/pruning/train_sparse_models)
       * internals not relevant to end-user (e.g. app and model developers) but relevant to
         others in ecosystem (e.g. hardware developers and other contributors).

4. Packaging and release:
   * releases are managed by the TensorFlow Model Optimization team. Work with
     them to produce releases.
   * auto-generated API docs.

5. Collaborative blog post (optional)
   * samples: [pruning
     API](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a)
     and [post-training integer quantization](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-post-training-integer-quantization-b4964a1ea9ba)
