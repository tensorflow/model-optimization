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

# TODO(b/185727163): switch to prebuilt Docker image to speed this up.

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

# Code under repo is checked out to
# ${KOKORO_ARTIFACTS_DIR}/github/tensorflow_model_optimization.
WORKDIR="${KOKORO_ARTIFACTS_DIR}/github/tensorflow_model_optimization"

docker build --tag tfmot \
  $WORKDIR/ci/kokoro/gcp_ubuntu

# Mount the checked out repository, make that the working directory and run
# ci/kokoro/build.sh from the repository, which runs all the unit tests.
docker run \
  --volume "${WORKDIR?}:${WORKDIR?}" \
  --workdir="${WORKDIR?}" \
  --rm \
  tfmot:latest \
  ci/kokoro/build.sh

# Kokoro will rsync this entire directory back to the executor orchestrating the
# build which takes forever and is totally useless.
sudo rm -rf "${KOKORO_ARTIFACTS_DIR?}"/*
