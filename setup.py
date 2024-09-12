from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import sys
import os
import glob
import shutil

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
        "simucloth.cy_simucloth.geom_utils",
        ["simucloth/cy_simucloth/geom_utils.pyx"],
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

# Copy .pyx files from cy_simucloth/src to simucloth/cy_simucloth
src_dir = 'cy_simucloth/src'
dest_dir = 'simucloth/cy_simucloth'
for ext in ('pyx', 'pxd'):
    for file in glob.glob(os.path.join(src_dir, f'*.{ext}')):
        shutil.copy(file, dest_dir)

try:
    setup(
        name="simucloth",
        packages=find_packages(),
        ext_modules=cythonize(ext_modules, annotate=True),
        zip_safe=False,
    )
except Exception as e:
    print(f"Error: {e}")

# Remove all .cpp, .pyx, .pxd and .c files in simucloth/cy_simucloth
for ext in ('cpp', 'c', 'pyx', 'pxd'):
    for file in glob.glob(f'simucloth/cy_simucloth/*.{ext}'):
        os.remove(file)

# Remove all .html files in cy_simucloth/perf_analysis
for file in glob.glob('cy_simucloth/perf_analysis/*.html'):
    os.remove(file)

# Move all .html files to simucloth/cy_simucloth/perf_analysis
perf_analysis_dir = 'cy_simucloth/perf_analysis'
os.makedirs(perf_analysis_dir, exist_ok=True)
for file in glob.glob('simucloth/cy_simucloth/*.html'):
    shutil.move(file, perf_analysis_dir)
