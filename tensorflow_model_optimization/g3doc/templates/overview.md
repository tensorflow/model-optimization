<!--
Template for per-technique overview page. Users use this page to determine
whether or not they want to try a technique. Once they've decided to try the technique,
they are guided to documentation on how to try the technique.

Remember that some people arriving this page will already have made the decision
to try the technique.

This page should minimize the background knowledge needed to understand the
text. This page targets end-users only, not contributors, hardware vendors, or
people who would be interested in the technique details. There will be a
separate page for that (TODO: determine location).

Regular brackets with text (e.g. {name}) should be replaced.

See https://www.tensorflow.org/model_optimization/guide/pruning for a good
example.
-->

# [Short Descriptor]

<!-- Describe purpose (to help users determine whether they want to use
technique) and then help navigate users who have previously decided they want
to use the technique to pages telling them how to use the technique. -->

This page provides an overview on {technique name} to help you determine how it
fits with your usecase.

*   To dive right into an end-to-end example, see the
    [{colab name} example]({link}).
*   To quickly find the APIs you need for your use case, see the
    [{technique name} comprehensive guide]({link}).

## Overview

{High level description with a few sentences}.

<!-- Short Pitch: Summary of Results Section -->

This technique brings improvements via {one or both of: model compression and
latency reduction}. {Sample: we see up to 4x improvements in model compression
and 2-3x latency reduction for vision applications and ... for speech
applications.}

<!--
Users can use this to consider the tradeoffs of using stable techniques versus
bleeding-edge techniques for additional gains.

Note: be careful about including non-public information or including names
of applications that haven't agreed to include their names.

If it's relevant, distinguish the technique and the code. E.g. The code is used
in A and B. The general technique is additionally used in C and D.
-->

The technique and code is used in production in {Applications X, Y, and Z}.

### API compatibility

<!--
Tell users what APIs we are currently compatible with. APIs include:
- Model building (e.g. tf.keras with Sequential/Functional/Subclassed, tf.layers)
- TensorFlow versions (e.g. TF 1.XX for versions 1.YY+, TF 2.XX) (TODO: have a table if
  this varies depending on model optimization pip version).
- TensorFlow execution mode: (e.g. graph mode, eager mode)
- Distributed training: (e.g. tf.distribute vs tf.estimator, synchronous vs asynchronous)

While the pitch helps users decide if the technique is worth their investment
(e.g. I need 3x improvements and 2x is not enough), this section allows users to
decide if they can or want to use the technique now.
-->

Users can apply {technique name} with the following APIs:

*   {Model building: tf.keras with only Sequential and Functional models} ...

<!-- Create Github issues and link to them.-->

It is on our roadmap to add support in the following areas:

*   {Model building: Subclassed support} ...

<!-- Optional section: consider additional factors such as model and layer
coverage, hardware coverage and links to docs describing the benefits to those hardware
(the specific benefits should live outside tensorflow.org/model_optimization), and more.

If the user can reap the techniques' benefits, almost regardless of model or hardware,
then the reason for this section diminishes.
-->

### General support matrix

Support is available in the following areas:

*   {Model coverage: models with layers X, Y, Z} ...

<!-- Create Github issues and link to them.-->

It is on our roadmap to add support in the following areas:

*   {Model coverage: extended coverage for usecases X and Y} ...

## Results

<!--
1. Subdivide into subsections for each mode of usage, if there is that
   distinction.

Example:

### Model compression only
[Table]

### Model compression and latency reduction
[Table]

2. If reasonable, for each subsection, have multiple tables for each use case (e.g. image classification, speech recognition).

3. If significant configuration changes are necessary for different models,
   consider including that information also as a detail after the tables.
   In the future, these models may be reproducible in TensorFlow Official Models
   and the configuration would be checked in.
-->

### Image classification

<!-- The critical hyperparameter is something that significantly affects the
accuracy and optimization metrics. An optimization metric is something like
latency or model storage size -->

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Baseline {Accuracy Metric}</th>
      <th>{Accuracy Metric} with {technique} </th>
      <th>Baseline {Optimization Metric} </th>
      <th>{Optimization Metric} with {technique} </th>
      <th>{Critical Hyperparameter} </th>
    </tr>
    <tr>
      <td>{MobilenetV2 224}</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
 </table>
</figure>

<!-- Optional: there may not be device-specific metrics -->

The {device-specific optimization metric} are generated on {device, ex: Pixel 2}
using {environment, e.g. single threaded large core}.

## Examples

<!-- Example may include one in TensorFlow Official Models.-->

In addition to the [{colab name} example]({link}), see the following examples:

*   {example 1}: [code]({link})

### Tips

<!-- Various training tips (e.g. hypertuning) corresponding to different use
cases. -->

## References

<!-- Discuss the connection of the implementation to primary research paper(s), with high
level summary of similarities and differences -->

{The ideas are based on A and B from papers X and Y.}

1.  **{Paper Name}** <br />
    {Authors}. <br />
    [{link}]({actual link}). {Accepted Conference}, {Year} <br />
