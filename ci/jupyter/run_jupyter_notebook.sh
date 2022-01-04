#!/usr/bin/env bash

# This code is heavily influenced by this script:
# https://github.com/google/iree/blob/1a049bc4f3dd1bde3734c9d1fafaac05ac593349/build_tools/testing/run_python_notebook.sh

# Exit immediately if there is a pipeline failure
# Enable tracing so errors are displayed
set -e
set -x

# Run under a virtual environment to isolate Python packages.
# This is done as jupyter notebooks can install python packages and we don't
# want to pollute the system packages.
python -m venv .notebook.venv --system-site-packages --clear
source .notebook.venv/bin/activate 2> /dev/null

# Set a trap so the virtual environment is deactivated if the script fails
trap "deactivate 2> /dev/null" EXIT

# Install the packages required for executing jupyter from the virtual environment
pip install --quiet \
  jupyter nbconvert ipykernel setuptools==59.5.0

export TF_CPP_MIN_LOG_LEVEL=2

# Run the notebook and pass the output to /dev/null
# This prevents the notebook being written to the filesystem
jupyter nbconvert --to notebook --execute $1 --stdout > /dev/null
