# from distutils.core import setup, Extension
# import numpy.distutils.misc_util

# # Adding OpenCV to project
# # ************************

# # Adding sources of the project
# # *****************************

# SOURCES = ["../cpp_utils/cloud/cloud.cpp",
#              "neighbors/neighbors.cpp",
#              "wrapper.cpp"]

# module = Extension(name="radius_neighbors",
#                     sources=SOURCES,
#                     extra_compile_args=['-std=c++11',
#                                         '-D_GLIBCXX_USE_CXX11_ABI=0'])


# setup(ext_modules=[module], include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs())

from setuptools import setup, Extension
import numpy as np
import os

# Source files
SOURCES = [
    os.path.join("..", "cpp_utils", "cloud", "cloud.cpp"),
    os.path.join("neighbors", "neighbors.cpp"),
    "wrapper.cpp"
]

module = Extension(
    "radius_neighbors",
    sources=SOURCES,
    include_dirs=[
        np.get_include(),
        os.path.abspath(".."), 
        os.path.abspath("../cpp_utils"),
        os.path.abspath("../cpp_utils/cloud"),
        os.path.abspath("../cpp_utils/grid"),
    ],
    extra_compile_args=[
        "-std=c++11",
        "-O3",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ]
)

setup(
    name="radius_neighbors",
    ext_modules=[module]
)









