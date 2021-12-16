#!/bin/bash

# Copyright 2020 Google LLC
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
# This script targets building for continuous integration. It can
# be used to reproduce errors locally by modifying WORKDIR to be
# the top-level directory of the checked out TFMOT Github repository.

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

# The TFMOT Git repository is checked out here.
DEFAULT_REPO_DIR="${KOKORO_ARTIFACTS_DIR}/github/tensorflow_model_optimization"
GIT_REPO_DIR="${GIT_REPO_DIR:-$DEFAULT_REPO_DIR}"


cleanup() {
  # Collect the test logs.
  docker exec tfmot find \
    -L "bazel-testlogs" \
    \( -name "test.log" -o -name "test.xml" \) \
    -exec cp --parents {} "${KOKORO_ARTIFACTS_DIR}" \;

  # Rename test.xml to sponge_log.xml so they show up in Sponge.
   docker exec tfmot find "${KOKORO_ARTIFACTS_DIR}/bazel-testlogs" \
    -type f \
    -name test.xml \
    -execdir mv "{}" sponge_log.xml \;

  # Rename test.log to sponge_log.log so they show up in Sponge.
  docker exec tfmot find "${KOKORO_ARTIFACTS_DIR}/bazel-testlogs" \
    -type f \
    -name test.log \
    -execdir mv "{}" sponge_log.log \;

  # Stop the container
  docker stop tfmot
}

# Build the Docker image.
# TODO(b/185727163): switch to prebuilt Docker image to speed this up.
docker build --tag tfmot \
  "${GIT_REPO_DIR}/ci/kokoro/gcp_ubuntu"

# Start a Docker container in the background.
# The Kokoro artitifacts directory is mounted and the work directory
# conveniently set to the TFMOT Git repository.
docker run \
  -it \
  -d \
  --rm \
  --name tfmot \
  --volume "${KOKORO_ARTIFACTS_DIR}:${KOKORO_ARTIFACTS_DIR}" \
  --workdir="${GIT_REPO_DIR}" \
  tfmot:latest \
  bash

# On exit: collect the test logs and stop the container.
trap cleanup EXIT

# Run the tests inside the container,
docker exec tfmot "${GIT_REPO_DIR}/ci/kokoro/build.sh"
