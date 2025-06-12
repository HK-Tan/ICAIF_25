# setup.py
import sys
from setuptools import setup, Extension
import pybind11
import os
# --- Helper function to find Eigen headers ---
def find_eigen_path():
    """Find the path to the Eigen headers installed by pip."""
    import site
    for site_path in site.getsitepackages():
        eigen_path = os.path.join(site_path, 'eigen')
        if os.path.exists(eigen_path):
            print(f"Found Eigen at: {eigen_path}")
            return eigen_path
    raise RuntimeError("Could not find Eigen headers. Please install with 'pip install eigen'.")
eigen_include_path = find_eigen_path()
# --- Define compiler arguments ---
extra_compile_args = ['-O3', '-DNDEBUG'] # High optimization, disable asserts
if sys.platform == 'win32':
    extra_compile_args = ['/O2', '/DNDEBUG'] # MSVC equivalent
else:
    extra_compile_args.extend(['-std=c++17', '-fvisibility=hidden'])
ext_modules = [
    Extension(
        # This becomes the name of the .so/.pyd file
        'cpp_rls_filter',
        ['./bindings.cpp'],
        include_dirs=[
            pybind11.get_include(),
            eigen_include_path,
            "./eigen-3.4.0/",
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
]
setup(
    name='cpp_rls_filter',
    version='1.0',
    author='Your Name',
    description='C++ RLS filter with Python bindings',
    ext_modules=ext_modules,
    zip_safe=False,
)
