import logging
import subprocess
import unittest

from pathlib import Path
from unittest import TestCase

logging.basicConfig(level=logging.INFO)

class JupyterNotebookTests(TestCase):
    """Test class for jupyter notebooks"""

    def test_notebooks(self):
        """Test function that generates a sub-test for each guide/clustering notebook"""
        script_path = Path('ci/jupyter/run_jupyter_notebook.sh')
        notebook_folders = [
            Path('tensorflow_model_optimization/g3doc/guide/clustering'),
            Path('tensorflow_model_optimization/g3doc/guide/combine')
        ]

        # Run each notebook as a subtest
        for folder in notebook_folders:
            for path in folder.glob('*.ipynb'):
                logging.info('Running the notebook {}'.format(path.name))

                with self.subTest(notebook=path.name):
                    completed_process = subprocess.run([script_path.absolute(), path.absolute()])
                    self.assertEqual(completed_process.returncode, 0)

if __name__ == '__main__':
    unittest.main()
