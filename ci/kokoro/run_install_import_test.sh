#!/bin/bash

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# A smoke test that installs TFMOT with `setup.py install` and then tries to
# import it.
#

# Make Bash more strict, for easier debugging.
set -e  # Exit on the first error.
set -u  # Treat unset variables as error.
set -o pipefail  # Treat the failure of a command in a pipeline as error.

# Display commands being run.
# WARNING: please only enable 'set -x' if necessary for debugging, and be very
#  careful if you handle credentials (e.g. from Keystore) with 'set -x':
#  statements like "export VAR=$(cat /tmp/keystore/credentials)" will result in
#  the credentials being printed in build logs.
#  Additionally, recursive invocation with credentials as command-line
#  parameters, will print the full command, with credentials, in the build logs.
# set -x


# Create an isolated Python environment.
mkdir venv
virtualenv ./venv
source ./venv/bin/activate

# Install Tensorflow, then TFMOT from source.
pip install tensorflow
pip install --requirement "requirements.txt"
python setup.py install

# Go to another directory and try to import TFMOT.
mkdir /tmp/my_project
pushd /tmp/my_project
python -c "import tensorflow_model_optimization"
