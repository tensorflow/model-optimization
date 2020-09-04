This document is for maintainers.

It is a work in progress and suggestions are welcome.

# Triaging
A issue or PR should only be assigned to someone if they would be actively working on
it or would be working on it shortly. This avoids the scenario when we think
everything is being addressed when that is not the case, particularly when people are
busy, which may cause issues to invisibly languish.

The [maintainer for a particular
technique](https://github.com/tensorflow/model-optimization#maintainers) is
responsible for that technique.

Github PRs
- The maintainers should watch out for PRs, filtered by the corresponding
  technique label. For example, the label for quantization-aware training is
  [here](https://github.com/tensorflow/model-optimization/pulls?q=is%3Apr+is%3Aopen+label%3Atechnique%3Aqat+).
  This labeling is automated
  [here](https://github.com/tensorflow/model-optimization/blob/master/.github/workflows/labeler.yml).

Github Issues
- The maintainers should skim through issues and attach the appropriate technique label. This has not been automated yet.
- Use other labels as needed, according to [CONTRIBUTING.md](https://github.com/tensorflow/model-optimization/blob/master/CONTRIBUTING.md).

# Code Reviews
Ensure that contributions adhere to the guidelines suggested in
[CONTRIBUTING.md](https://github.com/tensorflow/model-optimization/blob/master/CONTRIBUTING.md).
Comments that are frequently made in PRs should be added there.

If there are public API changes, please attach the ["api-review"
label](https://github.com/tensorflow/model-optimization/pulls?q=is%3Apr+is%3Aopen+label%3Aapi-review)
for a closer look from the TFMOT team.

Once you think a PR is ready to merge, apply the ["ready to pull"
label](https://github.com/tensorflow/model-optimization/pulls?q=is%3Apr+is%3Aopen+label%3A%22ready+to+pull%22)
and someone will merge it.
