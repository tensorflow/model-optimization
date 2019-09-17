# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tool to generate external api_docs for tensorflow_model_optimization.

Note:
  If duplicate or spurious docs are generated (e.g. internal names), consider
  blacklisting them via the `private_map` argument below.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# import g3
from absl import app
from absl import flags
from tensorflow_docs.api_generator import generate_lib

# import g3.third_party.tensorflow_model_optimization as tfmot


flags.DEFINE_string("output_dir", "/tmp/model_optimization_api",
                    "Where to output the docs")

flags.DEFINE_string(
    "code_url_prefix",
    ("https://github.com/tensorflow/model-optimization/blob/master/"
     "tensorflow_model_optimization"),
    "The url prefix for links to code.")

flags.DEFINE_bool("search_hints", True,
                  "Include metadata search hints in the generated files")

flags.DEFINE_string("site_path", "model_optimization/api_docs/python",
                    "Path prefix in the _toc.yaml")

FLAGS = flags.FLAGS


def main(unused_argv):
  doc_generator = generate_lib.DocGenerator(
      root_title="TensorFlow Model Optimization",
      py_modules=[("tfmot", tfmot)],
      base_dir=os.path.dirname(tfmot.__file__),
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      private_map={
          "tfmot.sparsity.keras": [
              # List of internal classes which get exposed when imported.
              "InputLayer", "custom_object_scope", "pruning_sched",
              "pruning_wrapper", "absolute_import", "division", "print_function"
          ]
      },
  )

  doc_generator.build(output_dir=FLAGS.output_dir)


if __name__ == "__main__":
  app.run(main)
