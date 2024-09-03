import sys
import numpy
from setuptools import setup
from mypyc.build import mypycify

extra_compile_args = ['/MD'] if sys.platform == 'win32' else []

ext_modules = mypycify(
    ['cloth_simulation.py'],
    opt_level="3",
)

# Manually add extra_compile_args and include_dirs to each Extension
for ext in ext_modules:
    ext.extra_compile_args = extra_compile_args
    ext.include_dirs.append(numpy.get_include())

setup(
    name='cloth_simulation',
    ext_modules=ext_modules,
    install_requires=['numpy'],
    packages=[''],
)