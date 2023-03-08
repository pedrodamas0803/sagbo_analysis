import pathlib
import sys

import setuptools

current_folder = pathlib.Path(__file__).parent

setuptools.setup(
    name='sagbo_analysis',
    version='0.0.1',
    author='Pedro D. Resende',
    author_email='pedro.damas-resende@esrf.fr',
    description='wrapper to facilitate my life analysing sagbo in-situ tests'
)
