from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import sys

shared_ext_kwargs = {
    'extra_compile_args': ['/O2'] if sys.platform == 'win32' else ['-O3'],
    'language': 'c++'
}

np_ext_kwargs = {
    "include_dirs": [np.get_include()],
    "define_macros": [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
}

ext_modules = [
    Extension(
        "simucloth.cy_simucloth.sim_objects",
        ["simucloth/cy_simucloth/sim_objects.pyx"],
        **shared_ext_kwargs
    ),
    Extension(
        "simucloth.cy_simucloth.perlin_noise",
        ["simucloth/cy_simucloth/perlin_noise.pyx"],
        **shared_ext_kwargs
    ),
    Extension(
        "simucloth.cy_simucloth.wind",
        ["simucloth/cy_simucloth/wind.pyx"],
        **np_ext_kwargs,
        **shared_ext_kwargs
    ),
    Extension(
        "simucloth.cy_simucloth.cloth_simulation",
        ["simucloth/cy_simucloth/cloth_simulation.pyx"],
        **np_ext_kwargs,
        **shared_ext_kwargs
    )
]

setup(
    name="simucloth",
    packages=find_packages(),
    ext_modules=cythonize(ext_modules, annotate=True),
    zip_safe=False,
)