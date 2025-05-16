from setuptools import setup
from Cython.Build import cythonize
import sys

setup(
    name='gui_main',
    ext_modules=cythonize("gui_main.py", compiler_directives={"language_level": "3"}),
    zip_safe=False,
)
