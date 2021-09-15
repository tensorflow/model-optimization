# Jupyter Notebook CI
The `test_jupyter_notebooks.py` program runs Jupyter notebooks in the 
`tensorflow_model_optimisation/g3doc/guide/clustering` as sub tests. This intended to be used
as part of a CI system.

## Running Locally
To run this script locally using Docker, run the following commands from the root directory of this repository:
1. `docker pull tensorflow/tensorflow`
2. `docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow bash`
3. `apt-get install python3-venv`
4. `pip install --upgrade pip tensorflow-model-optimization`
5. `python ci/jupyter/test_jupyter_notebooks.py`
