# Keras Pruning API

| Status        | (Proposed / Accepted / Implemented / Obsolete)       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/model-optimization/pull/NNN) (update when you have the PR #)|
| **Author(s)** | Sample Name (me@email.com)                           |
| **Sponsor**   | Sample Sponsor (whomever@email.com)                  |
| **Updated**   | 2020-02-25                                           |

We propose to implement the following technique on top of the Keras API.

## Motivation and Overview

Magnitude-based weight pruning gradually zeroes out model weights during the
training process to achieve model sparsity. Sparse models are easier to
compress, and we can skip the zeroes during inference for latency improvements.

This technique brings improvements via model compression. Further framework
support for  this technique in TF can result in latency improvements. We've
seen up to 6x improvements in model compression with minimal loss of accuracy.

The technique is being evaluated in various speech applications, such as
speech recognition and text-to-speech, and has been experimented on across
various vision and translation models.

## Results

### Image Classification

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Non-sparse Top-1 Accuracy </th>
      <th>Sparse Accuracy </th>
      <th>Sparsity </th>
    </tr>
    <tr>
      <td rowspan=3>InceptionV3</td>
      <td rowspan=3>78.1%</td>
      <td>78.0%</td>
      <td>50%</td>
    </tr>
    <tr>
      <td>76.1%</td><td>75%</td>
    </tr>
    <tr>
      <td>74.6%</td><td>87.5%</td>
    </tr>
    <tr>
      <td>MobilenetV1 224</td><td>71.04%</td><td>70.84%</td><td>50%</td>
    </tr>
 </table>
</figure>

The models were tested on Imagenet.

### Translation

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Non-sparse BLEU </th>
      <th>Sparse BLEU </th>
      <th>Sparsity </th>
    </tr>
    <tr>
      <td rowspan=3>GNMT EN-DE</td>
      <td rowspan=3>26.77</td>
      <td>26.86</td>
      <td>80% </td>
    </tr>
    <tr>
      <td>26.52</td><td>85%</td>
    </tr>
    <tr>
      <td>26.19</td><td>90%</td>
    </tr>
    <tr>
      <td rowspan=3>GNMT DE-EN</td>
      <td rowspan=3>29.47</td>
      <td>29.50</td>
      <td>80% </td>
    </tr>
    <tr>
      <td>29.24</td><td>85%</td>
    </tr>
    <tr>
      <td>28.81</td><td>90%</td>
    </tr>
 </table>
</figure>

The models use WMT16 German and English dataset with news-test2013 as the dev
set and news-test2015 as the test set.

For background, see *To prune, or not to prune: exploring the efficacy of
pruning for model compression* [[paper](https://arxiv.org/pdf/1710.01878.pdf)].

## Contribution

<!-- TODO(tfmot): decide if it makes sense to only have proposals when a clear
owner and implementer exists. Yes would be reasonable. -->
The Model Optimization team from TensorFlow (initially X, Y, Z) will be building the technique with the intention of being a
production-level owner.


