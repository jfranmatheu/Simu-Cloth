from setuptools import setup
from mypyc.build import mypycify

setup(
    name='cloth_simulation',
    ext_modules=mypycify([
        'cloth_simulation.py',
    ]),
    packages=[''],
)