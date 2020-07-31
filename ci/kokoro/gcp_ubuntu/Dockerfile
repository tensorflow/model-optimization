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

# An image for building prerequisites for testing TensorFlow Model Optimization
# on ubuntu.
#
# TODO(tfmot): generalize to different versions of TensorFlow to
# run CI against.
#
#  Build as follows:
#  docker build --tag={TAG} /path/to/directory/with/this/file
#
#  Run interactively as follows, where WORKDIR is the root directory
#  of the tensorflow model optimization Github repository.
#
#  docker run -it --volume "${WORKDIR}:/usr/src/tfmot" {TAG}

FROM ubuntu:18.04
WORKDIR /usr/src/tfmot

RUN apt-get update -y

# Install Python 3.
#
# Using Python 3.7 since 3.8 is not compatible with TensorFlow
# versions <= 2.1.
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.7
ENV PYTHON_BIN /usr/bin/python3.7

# Install Bazel.
# https://docs.bazel.build/versions/master/install-ubuntu.html
ARG BAZEL_VERSION=3.4.1
RUN apt-get install -y unzip zip wget \
   && wget "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION?}/bazel-${BAZEL_VERSION?}-installer-linux-x86_64.sh" \
   && chmod +x "bazel-${BAZEL_VERSION?}-installer-linux-x86_64.sh" \
   && "./bazel-${BAZEL_VERSION?}-installer-linux-x86_64.sh" --user \
   && rm "bazel-${BAZEL_VERSION?}-installer-linux-x86_64.sh"
ENV PATH "/root/bin:${PATH}"

# Install clang, which is required by bazel during building TFMOT.
RUN apt-get install -y clang
ENV CC clang

# Install pip.
RUN apt-get install -y python3-pip
# Upgrade pip3 to be able to detect additional TensorFlow versions.
RUN pip3 install --upgrade pip

# Install latest TensorFlow 2.0.X.
RUN pip3 install tensorflow==2.0.0
