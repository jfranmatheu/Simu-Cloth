from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

ext_modules = [
    Extension(
        "simucloth.cy_simucloth.cloth_simulation",
        ["simucloth/cy_simucloth/cloth_simulation.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        extra_compile_args=['/O2'] if sys.platform == 'win32' else ['-O3'],
        language="c++"
    )
]

setup(
    name="cy_simucloth",
    ext_modules=cythonize(ext_modules, annotate=True),
    zip_safe=False,
)